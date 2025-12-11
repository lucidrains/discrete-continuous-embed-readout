import pytest
param = pytest.mark.parametrize

# i/o to attention

import torch
from x_transformers import Decoder

# tests

from discrete_continuous_embed_readout import (
    Embed,
    Readout,
    EmbedAndReadout
)

def test_discrete_autoregressive():

    token_ids = torch.randint(0, 20000, (2, 64))

    past, future = token_ids[:, :-1], token_ids[:, 1:]

    embed, readout = EmbedAndReadout(512, num_discrete = 20_000)

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()

    logits = readout(attended)

    assert logits.shape == (2, 63, 20000)

    sampled = readout.sample(logits)
    assert sampled.shape == (2, 63)

@param('pred_log_var', (False, True))
@param('continuous_norm', (False, True))
def test_continuous_autoregressive(
    pred_log_var,
    continuous_norm
):

    tokens = torch.randn(2, 64, 5)

    past, future = tokens[:, :-1], tokens[:, 1:]

    # maybe handle norm

    continuous_mean_std = None

    if continuous_norm:
        continuous_mean_std = torch.ones((5, 2))

    embed, readout = EmbedAndReadout(
        512,
        num_continuous = 5,
        continuous_mean_std = continuous_mean_std,
        readout_kwargs = dict(
            continuous_log_var_embed = pred_log_var,
        )
    )

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()

    dist = readout(attended)

    assert dist.shape == (2, 63, 5, 2) if pred_log_var else (2, 63, 5)

    if pred_log_var:
        sampled = readout.sample(dist)
        assert sampled.shape == (2, 63, 5)

def test_discrete_continuous_autoregressive():

    continuous_tokens = torch.randn(2, 64, 5)

    discrete_token_ids = torch.randint(0, 2000, (2, 64))

    past_discrete, future_discrete = discrete_token_ids[:, :-1], discrete_token_ids[:, 1:]

    past_continuous, future_continuous = continuous_tokens[:, :-1], continuous_tokens[:, 1:]

    embed, readout = EmbedAndReadout(512, num_discrete = 20_000, num_continuous = 5)

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    tokens = embed((past_discrete, past_continuous))

    attended = attn(tokens)

    loss = readout(attended, (future_discrete, future_continuous), return_loss = True)

    loss.backward()

    discrete_logits, continuous_mu_log_var = readout(attended)

    assert discrete_logits.shape == (2, 63, 20_000)
    assert continuous_mu_log_var.shape == (2, 63, 5, 2)

    all_logits = readout(attended)
    sampled_discrete, sampled_continuous = readout.sample(all_logits)

    assert sampled_discrete.shape == (2, 63)
    assert sampled_continuous.shape == (2, 63, 5)

def test_multi_discrete_autoregressive():

    token_ids = torch.randint(0, 500, (2, 64, 2))

    past, future = token_ids[:, :-1], token_ids[:, 1:]

    embed, readout = EmbedAndReadout(512, num_discrete = (500, 500))

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()

    logits = readout(attended)

    assert all([logit.shape == (2, 63, 500) for logit in logits])

    sampled = readout.sample(logits)
    assert sampled.shape == (2, 63, 2)