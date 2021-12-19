import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math, copy, time
from typing import NoReturn, Optional, Callable, Generator, Iterable, List
import matplotlib.pyplot as plt
import seaborn
from collections import OrderedDict
import pickle
import gensim
import sklearn

class EncoderDecoder(nn.Module):

  def __init__(self,
               encoder: nn.Module,
               decoder: nn.Module,
               src_embed: nn.Module) -> NoReturn:
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed

  def forward(self,
              src: torch.Tensor,
              src_mask: torch.Tensor) -> torch.Tensor:
    return self.decode(self.encode(src, src_mask))

  def encode(self,
             src: torch.Tensor,
             src_mask: torch.Tensor) -> torch.Tensor:
    return self.encoder(self.src_embed(src), src_mask)

  def decode(self,
             memory: torch.Tensor):
    return self.decoder(memory)


class Encoder(nn.Module):

  def __init__(self, encoder_layer: nn.Module, n_layer: int) -> NoReturn:
    super(Encoder, self).__init__()
    self.layers = clones(encoder_layer, n_layer)
    self.norm_layer = nn.LayerNorm(encoder_layer.size)

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x, mask)
    x = self.norm_layer(x) # because of preLN
    return x


class EncoderLayer(nn.Module):

  def __init__(self,
               size: int,
               multi_head_attention_layer: nn.Module,
               position_wise_feed_forward_layer: nn.Module,
               dropout: float) -> NoReturn:
    super(EncoderLayer, self).__init__()
    self.multi_head_attention_layer = multi_head_attention_layer
    self.attention_residual_connection_layer = ResidualConnectionLayer(size,
                                                                       dropout)
    #self.attention_residual_connection_layer = SublayerConnection(size, dropout)
    self.position_wise_feed_forward_layer = position_wise_feed_forward_layer
    self.ff_residual_connection_layer = ResidualConnectionLayer(size,
                                                                dropout)
    #self.ff_residual_connection_layer = SublayerConnection(size, dropout)
    self.size = size
  
  def forward(self,
              x: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    out = self.attention_residual_connection_layer(
        x, lambda x: self.multi_head_attention_layer(x, x, x, mask)
    )
    out = self.ff_residual_connection_layer(
        out, self.position_wise_feed_forward_layer
    )
    return out


class MultiHeadAttentionLayer(nn.Module):
  def __init__(self, d_model: int, h: int) -> NoReturn:
    super(MultiHeadAttentionLayer, self).__init__()
    assert d_model % h == 0

    self.d_model = d_model
    self.d_k = d_model // h
    self.h = h
    self.query_fc_layer = nn.Linear(d_model, d_model)
    self.key_fc_layer = nn.Linear(d_model, d_model)
    self.value_fc_layer = nn.Linear(d_model, d_model)
    self.final_fc_layer = nn.Linear(d_model, d_model)

    self.attention_prob = None

  def calculate_attention(self,
                          query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          mask: torch.Tensor):
    d_k = key.size(-1)
    attention_score = torch.matmul(query, key.transpose(-2, -1))
    attention_score = attention_score / math.sqrt(d_k)
    if mask is not None:
      attention_score = attention_score.masked_fill(mask == 0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1)
    out = torch.matmul(attention_prob, value)
    return out, attention_prob

  def forward(self,
              query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    n_batch = query.shape[0]

    def transform(x: torch.Tensor, fc_layer: nn.Module) -> torch.Tensor:
      out = fc_layer(x)
      out = out.view(n_batch, -1, self.h, self.d_k)
      out = out.transpose(1, 2)
      return out

    query = transform(query, self.query_fc_layer)
    key = transform(key, self.key_fc_layer)
    value = transform(value, self.value_fc_layer)

    if mask is not None:
      mask = mask.unsqueeze(1)

    out, self.attention_prob = self.calculate_attention(query, key, value, mask)
    out = out.transpose(1, 2)
    out = out.contiguous().view(n_batch, -1, self.d_model)
    out = self.final_fc_layer(out)
    return out


class PositionWiseFeedForwardLayer(nn.Module):
  def __init__(self, d_model: int, d_ff: int) -> NoReturn:
    super(PositionWiseFeedForwardLayer, self).__init__()
    self.first_fc_layer = nn.Linear(d_model, d_ff)
    self.second_fc_layer = nn.Linear(d_ff, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.first_fc_layer(x)
    out = F.relu(out)
    out = self.second_fc_layer(out)
    return out


class ResidualConnectionLayer(nn.Module):
  def __init__(self, size: int, dropout: float) -> NoReturn:
    super(ResidualConnectionLayer, self).__init__()
    self.norm_layer = nn.LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self,
              x: torch.Tensor,
              sub_layer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    #return self.norm_layer(x + self.dropout(sub_layer(x))) # postLN
    return x + self.dropout(sub_layer(self.norm_layer(x))) # preLN


class Decoder(nn.Module):
  def __init__(self, d_model: int, n_seq: int, d_ff: int, n_classes: int, dropout: float) -> NoReturn:
    super(Decoder, self).__init__()
    self.first_fc_layer = nn.Linear(d_model * n_seq, d_ff)
    self.second_fc_layer = nn.Linear(d_ff, n_classes)
    self.dropout = nn.Dropout(dropout)

  def forward(self,
              memory: torch.Tensor) -> torch.Tensor:
    memory = memory.view(memory.size(0), -1)
    out = self.first_fc_layer(memory)
    out = F.relu(out)
    out = self.dropout(out)
    out = self.second_fc_layer(out)
    return F.log_softmax(out, dim=-1)


class PositionalEncoding(nn.Module):
  def __init__(self,
               d_model: int,
               dropout: float,
               max_len: int=5000) -> NoReturn:
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1) # max_len X 1
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor):
    x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
    return self.dropout(x)


def clones(module: nn.Module, N: int) -> nn.ModuleList:
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def make_model(N: int=6,
               d_model: int=512,
               n_seq: int=1000,
               d_ff: int=2048,
               d_ff2: int=128,
               n_classes: int=3,
               h: int=8,
               dropout: float=0.1) -> nn.Module:
  c = copy.deepcopy
  #attn = MultiHeadedAttention(h, d_model)
  attn = MultiHeadAttentionLayer(d_model, h)
  #ff = PositionwiseFeedForward(d_model, d_ff)
  ff = PositionWiseFeedForwardLayer(d_model, d_ff)
  rc = ResidualConnectionLayer(d_model, dropout)
  pe = PositionalEncoding(d_model, dropout)
  model = EncoderDecoder(
      # Encoder(EncodeLayer(c(attn), c(ff), c(rc)), N),
      Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
      # Decoder(DecoderLayer(c(attn), c(ff), c(rc)), N),
      Decoder(d_model, n_seq, d_ff2, n_classes, dropout),
      c(pe)
  )

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model


class Batch:

  def __init__(self,
               src: torch.Tensor,
               src_mask: torch.Tensor,
               trg: Optional[torch.Tensor]=None,
               pad: int=0) -> NoReturn:
    self.src = src
    self.src_mask = src_mask
    self.trg = trg
    self.ntokens = (self.src_mask != pad).sum()


def run_epoch(data_iter: Iterable[Batch],
              model: nn.Module,
              loss_compute: Callable[[torch.Tensor, torch.Tensor], float]
              ) -> float:
  start = time.time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  for i, batch in enumerate(data_iter):
    out = model.forward(batch.src, batch.src_mask)
    loss = loss_compute(out, batch.trg, batch.ntokens)
    total_loss += loss
    total_tokens += batch.ntokens
    tokens += batch.ntokens
    if i % 50 == 1:
      elapsed = time.time() - start
      print(f'Epoch Step: {i} Loss: {loss / batch.ntokens}, Tokens per Sec: {tokens / elapsed}')
      start = time.time()
      tokens = 0
  return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
  global max_src_in_batch, max_tgt_in_batch
  if count == 1:
    max_src_in_batch = 0
    max_tgt_in_batch = 0
  max_src_in_batch = max(max_src_in_batch, len(new.src))
  max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
  src_elements = count * max_src_in_batch
  tgt_elements = count * max_tgt_in_batch
  return max(src_elements, tgt_elements)


class NoamOpt:
  def __init__(self,
               model_size: int,
               factor: int,
               warmup: int,
               optimizer: torch.optim.Optimizer) -> NoReturn:
    self.optimizer = optimizer
    self._step = 0
    self.warmup = warmup
    self.factor = factor
    self.model_size = model_size
    self._rate = 0

  def step(self) -> NoReturn:
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()

  def rate(self, step: int=None) -> float:
    if step is None:
      step = self._step
    return self.factor * (self.model_size ** (-0.5) *
                          min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
def get_std_opt(model: nn.Module) -> NoamOpt:
  return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                 torch.optim.Adam(model.parameters(), lr=0,
                                  betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
  def __init__(self,
               size: int,
               smoothing: float=0.0) -> NoReturn:
    super(LabelSmoothing, self).__init__()
    self.criterion = nn.KLDivLoss(reduction='sum')
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing
    self.size = size
    self.true_dist = None

  def forward(self,
              x: torch.Tensor,
              target: torch.Tensor) -> float:
    assert x.size(1) == self.size
    true_dist = x.clone().detach().requires_grad_(False)
    true_dist.fill_(self.smoothing / (self.size - 1))
    true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
    # true_dist[:, self.padding_idx] = 0
    #mask = torch.nonzero(target == self.padding_idx)
    #if mask.dim() > 0:
    #  true_dist.index_fill_(0, mask.squeeze(), 0.0)
    self.true_dist = true_dist
    return self.criterion(x, true_dist)


class SimpleLossCompute:
  def __init__(self,
               criterion: nn.Module,
               opt: NoamOpt=None) -> NoReturn:
    self.criterion = criterion
    self.opt = opt

  def __call__(self, x, y, norm):
    #loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
    #                      y.contiguous().view(-1)) / norm
    loss = self.criterion(x, y) / norm
    loss.backward()
    if self.opt is not None:
      self.opt.step()
      self.opt.optimizer.zero_grad()
    return loss * norm
