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

    logits = readout(attended)

    assert logits.shape == (2, 63, 20000)

@param('pred_log_var', (False, True))
def test_continuous_autoregressive(
    pred_log_var
):

    tokens = torch.randn(2, 64, 5)

    past, future = tokens[:, :-1], tokens[:, 1:]

    embed = Embed(512, num_continuous = 5)

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    readout = Readout(512, num_continuous = 5, continuous_log_var_embed = pred_log_var)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()

    dist = readout(attended)

    assert dist.shape == (2, 63, 5, 2) if pred_log_var else (2, 63, 5)

def test_discrete_continuous_autoregressive():

    continuous_tokens = torch.randn(2, 64, 5)

    discrete_token_ids = torch.randint(0, 2000, (2, 64))

    past_discrete, future_discrete = discrete_token_ids[:, :-1], discrete_token_ids[:, 1:]

    past_continuous, future_continuous = continuous_tokens[:, :-1], continuous_tokens[:, 1:]

    embed = Embed(512, num_discrete = 20_000, num_continuous = 5)

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    readout = Readout(512, num_discrete = 20_000, num_continuous = 5)

    tokens = embed((past_discrete, past_continuous))

    attended = attn(tokens)

    loss = readout(attended, (future_discrete, future_continuous), return_loss = True)

    loss.backward()

    discrete_logits, continuous_mu_log_var = readout(attended)

    assert discrete_logits.shape == (2, 63, 20_000)
    assert continuous_mu_log_var.shape == (2, 63, 5, 2)
