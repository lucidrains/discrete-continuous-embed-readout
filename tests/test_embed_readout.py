import pytest
param = pytest.mark.parametrize

# i/o to attention

import torch
from x_transformers import Decoder

# tests

from discrete_continuous_embed_readout import (
    Embed,
    Readout
)

def test_discrete_autoregressive():

    token_ids = torch.randint(0, 20000, (2, 64))

    past, future = token_ids[:, :-1], token_ids[:, 1:]

    embed = Embed(512, num_discrete = 20_000)

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    readout = Readout(512, num_discrete = 20_000)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()

def test_continuous_autoregressive():

    tokens = torch.randn(2, 64, 5)

    past, future = tokens[:, :-1], tokens[:, 1:]

    embed = Embed(512, num_continuous = 5)

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    readout = Readout(512, num_continuous = 5)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()
