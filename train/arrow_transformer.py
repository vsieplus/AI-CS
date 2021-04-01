# implementation of an 'arrow' transformer model with pytorch for step selection
# architecture ideas drawn from
#   transformers/other attention based mechanisms in nlp: arxiv.org/abs/1706.03762
#   music transformer: arxiv.org/abs/1809.04281
# Implementation adapted from https://github.com/jason9693/MusicTransformer-pytorch

# Model Description (see paper for more detailed/formal definition)

import math
import numpy as np

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

def sequence_mask(lengths, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def get_masked_with_pad_tensor(size, src, trg, pad_token):
    src = src[:, None, None, :]
    trg = trg[:, None, None, :]
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = torch.equal(src, src_pad_tensor)
    trg_mask = torch.equal(src, src_pad_tensor)
    if trg is not None:
        trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
        dec_trg_mask = trg == trg_pad_tensor
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = ~sequence_mask(torch.arange(1, size+1).to(trg.device), size)
        # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
        look_ahead_mask = dec_trg_mask | seq_mask

    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask

class ArrowTransformer(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_layers, max_seq, pad_token, conditioning=False, dropout=0.2):
        super().__init__()

        self.infer = False

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq = max_seq
        self.pad_token = pad_token

        self.Decoder = Encoder(self.num_layers, self.embed_dim, self.vocab_size, dropout, self.max_seq)
        self.fc = nn.Linear(self.embed_dim, self.vocab_size)

        self.conditioning = conditioning

    def forward(self, x, clstm_hiddens, length=None):
        # TODO implement conditioning with clstm_hiddens

        if not self.infer:
            _, _, mask = get_masked_with_pad_tensor(self.max_seq, x, x, self.pad_token)
            decoded, w = self.Decoder(x, mask)
            fc = self.fc(decoded)

            return (fc.contiguous(), [weight.contiguous() for weight in w])
        else:
            return self.generate(x, length, None).continguous().tolist()

    def generate(self, prior, length=2048):
        decoded = prior
        outputs = prior

        for i in range(length):
            _, _, mask = get_masked_with_pad_tensor(decoded.size(1), decoded, decoded, self.pad_token)

            result, _ = self.Decoder(decoded, mask)
            result = self.fc(result)
            result = result.softmax(dim=-1)

            pdf = dist.OneHotCategorical(probs=result[:, -1])
            result = pdf.sample().argmax(-1).unsqueeze(-1)

            decoded = torch.cat((decoded, result), dim=-1)
            outputs = torch.cat((outputs, result), dim=-1)

            del mask

        outputs = outputs[0]

        return outputs

    def test(self):
        self.eval()
        self.infer = True

class PositionEmbedding(nn.Module):
    """Encode the position of sequence elements by adding a position embedding"""
    def __init__(self, embed_dim, max_seq_len=2048):
        super().__init__()

        # shape: (1, max_seq_len, embed_dim)
        self.embed = np.array([[
            [
                math.sin(
                    position * math.exp(-math.log(10000) * i / embed_dim) * math.exp(
                        math.log(10000) / embed_dim * (i % 2)) + 0.5 * math.pi * (i % 2)
                )
                for i in range(embed_dim)
            ]
            for position in range(max_seq_len)
        ]])

    def forward(self, x):
        # x is shape (batch, max_seq_len, embed_dim)
        x = x + torch.from_numpy(self.embed[:, :x.size(1), :].to(x.device, dtype=x.dtype))
        return x

class RelativeGlobalAttention(nn.Module):
    """Compute self-attention using relative position information"""
    def __init__(self, heads=4, dim=256, add_embed=False, max_seq_len=2048):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.add_embed = add_embed
        self.max_seq_len = max_seq_len

        # dimension of each head
        self.head_dim = dim // heads

        # relative position embedding
        self.E = torch.randn((max_seq_len, int(self.head_dim)), requires_grad=False)

        # query, key, value matrices
        self.Wq = nn.Linear(self.dim, self.dim)
        self.Wk = nn.Linear(self.dim, self.dim)
        self.Wv = nn.Linear(self.dim, self.dim)

        self.fc = nn.Linear(self.dim, self.dim)

    def __forward__(self, inputs, mask=None, **kwargs):
        """
        inputs: list of tensors [q, k, v]
        mask (optional): to mask certain inputs (for batching)
        """
        q, k, v = inputs

        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3) # [batch, head, seq_len, head_dim]

        # similar for key/value:
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        # extract only the necessary embeddings for the given input length
        start = max(0, self.max_seq_len - self.len_q)
        E = self.E[start:, :]

        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        
        logits = QKt + Srel
        logits = logits / math.sqrt(self.head_dim)

        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        attn_weights = F.softmax(logits, dim=-1)
        attn = torch.matmul(attn_weights, v)

        out = attn.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        # shape (batch, seq_len, dim)
        out = self.fc(out)

        return out, attn_weights

    def _skewing(self, tensor):
        """As described in Music Transformer, section 3.4.1"""
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.size(-1) - 1, qe.size(-1) - qe.size(-2) - 1, -1).to(device),
            qe.size(-1)
        )
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe

class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, h=16, additional=False, max_seq=2048):
        super().__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(heads=h, dim=d_model, max_seq_len=max_seq, add_embed=additional)

        self.FFN_pre = nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, w = self.rga([x, x, x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(ffn_out)

        return out2, w

class Encoder(nn.Module):
    def __init__(self, num_layers, dim_model, input_vocab_size, dropout=0.1, max_len=None):
        super().__init__()

        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=dim_model)
        self.pos_encoding = PositionEmbedding(self.dim_model, max_seq=max_len)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(dim_model, dropout, dim_model // 64, additional=False, max_seq=max_len)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def __forward__(self, x, mask=None):
        weights = []

        # [batch, seq, embed]
        x = self.embedding(x.to(torch.long))

        x *= math.sqrt(self.dim_model)

        x = self.pos_encoding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask)

        return x, weights