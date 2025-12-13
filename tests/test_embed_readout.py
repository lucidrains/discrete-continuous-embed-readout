import pytest
param = pytest.mark.parametrize

# i/o to attention

import torch
from x_transformers import Decoder

# tests

from discrete_continuous_embed_readout.discrete_continuous_embed_readout import (
    Embed,
    Readout,
    EmbedAndReadout,
    segmented_softmax,
    exists
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

    log_prob = readout.log_prob(logits, sampled)
    assert log_prob.shape == (2, 63)

    entropy = readout.entropy(logits)
    assert entropy.shape == (2, 63)

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

    discrete_loss, continuous_loss = readout(attended, (future_discrete, future_continuous), return_loss = True)

    (discrete_loss + 0.1 * continuous_loss).backward()

    discrete_logits, continuous_mu_log_var = readout(attended)

    assert discrete_logits.shape == (2, 63, 20_000)
    assert continuous_mu_log_var.shape == (2, 63, 5, 2)

    all_logits = readout(attended)
    sampled = readout.sample(all_logits)

    sampled_discrete, sampled_continuous = sampled
    assert sampled_discrete.shape == (2, 63)
    assert sampled_continuous.shape == (2, 63, 5)

    log_prob_discrete, log_prob_continuous = readout.log_prob(all_logits, sampled)

    assert log_prob_discrete.shape == (2, 63)
    assert log_prob_continuous.shape == (2, 63, 5)

    entropy_discrete, entropy_continuous = readout.entropy(all_logits)
    assert entropy_discrete.shape == (2, 63)
    assert entropy_continuous.shape == (2, 63, 5)

@param('use_parallel_multi_discrete', (False, True))
def test_multi_discrete_autoregressive(
    use_parallel_multi_discrete
):

    token_ids = torch.randint(0, 500, (2, 64, 2))

    past, future = token_ids[:, :-1], token_ids[:, 1:]

    embed, readout = EmbedAndReadout(
        512,
        num_discrete = (500, 500),
        use_parallel_multi_discrete = use_parallel_multi_discrete
    )

    attn = Decoder(dim = 512, depth = 1, rotary_pos_emb = True)

    tokens = embed(past)

    attended = attn(tokens)

    loss = readout(attended, future, return_loss = True)

    loss.backward()

    logits = readout(attended)

    assert all([logit.shape == (2, 63, 500) for logit in logits])

    sampled = readout.sample(logits)
    assert sampled.shape == (2, 63, 2)

    log_probs = readout.log_prob(logits, sampled)
    assert log_probs.shape == (2, 63, 2)

    entropy = readout.entropy(logits)
    assert entropy.shape == (2, 63, 2)

def test_multi_discrete_embed():

    token_ids = torch.randint(0, 500, (2, 64, 2))

    embed = Embed(512, num_discrete = (500, 500))

    embedded_groups = embed(token_ids, sum_discrete_sets = False)

    assert embedded_groups.shape == (2, 64, 2, 512)

def test_none():

    embed, readout = EmbedAndReadout(512, num_discrete = 20_000, return_only_discrete_or_continuous = False)

    logits, none = readout(torch.randn(2, 63, 512))

    assert none is None

    assert logits.shape == (2, 63, 20000)

    sampled = readout.sample(logits)
    assert sampled.shape == (2, 63)

    log_prob = readout.log_prob(logits, sampled)
    assert log_prob.shape == (2, 63)

    entropy = readout.entropy(logits)
    assert entropy.shape == (2, 63)

def test_segmented_softmax():
    dims = (5, 10, 3)

    logits = torch.randn(2, 63, sum(dims))

    probs_flat = segmented_softmax(logits, dims)

    probs_refs = [l.softmax(dim = -1) for l in logits.split(dims, dim = -1)]
    probs_refs_flat = torch.cat(probs_refs, dim = -1)

    assert torch.allclose(probs_flat, probs_refs_flat, atol = 1e-6)

def test_kl_div_parallel_equality():
    readout = Readout(512, num_discrete = (500, 1000, 500), use_parallel_multi_discrete = True)

    logits_true = [torch.randn(2, 63, 500), torch.randn(2, 63, 1000), torch.randn(2, 63, 500)]
    logits_pred = [torch.randn(2, 63, 500), torch.randn(2, 63, 1000), torch.randn(2, 63, 500)]

    kl_parallel = readout.kl_div_discrete(logits_true, logits_pred)

    readout.use_parallel_multi_discrete = False
    kl_sequential = readout.kl_div_discrete(logits_true, logits_pred)

    assert torch.allclose(kl_parallel, kl_sequential, atol = 1e-6)

def test_multiple_selectors():
    # 1. discrete AR (20000)
    # 2. continuous AR (5)
    # 3. mixed (20000 distinct from 1., 5 distinct from 2.)
    # 4. multi-discrete (500, 500)

    # 1. discrete AR config

    config_discrete_ar = [[i for i in range(20000)]]

    # 2. continuous AR config

    config_continuous_ar = [i for i in range(5)]

    # 3. mixed config

    config_mixed_discrete = [[i + 20000 for i in range(20000)]]
    config_mixed_continuous = [i + 5 for i in range(5)]

    # 4. multi-discrete config

    config_multi_discrete = [
        [i + 40000 for i in range(500)],
        [i + 40500 for i in range(500)]
    ]

    selectors = [
        config_discrete_ar,
        config_continuous_ar,
        (config_mixed_discrete, config_mixed_continuous),
        config_multi_discrete
    ]

    embed, readout = EmbedAndReadout(
        dim = 512,
        num_discrete = 41000,
        num_continuous = 10,
        selectors = selectors,
        readout_kwargs = dict(
            continuous_log_var_embed = True
        )
    )

    # 1. discrete AR

    token_ids = torch.randint(0, 20000, (2, 64))

    tokens = embed(token_ids, selector_index = 0)
    assert tokens.shape == (2, 64, 512)

    logits = readout(tokens, selector_index = 0)
    assert logits.shape == (2, 64, 20000)

    sampled = readout.sample(logits, selector_index = 0)
    assert sampled.shape == (2, 64)

    # 2. continuous AR

    continuous_input = torch.randn(2, 64, 5)

    tokens = embed(continuous_input, selector_index = 1)
    assert tokens.shape == (2, 64, 512)

    dist = readout(tokens, selector_index = 1)
    assert dist.shape == (2, 64, 5, 2)

    # 3. mixed

    discrete_inp = torch.randint(0, 20000, (2, 64))
    continuous_inp = torch.randn(2, 64, 5)

    tokens = embed((discrete_inp, continuous_inp), selector_index = 2)
    assert tokens.shape == (2, 64, 512)

    out = readout(tokens, selector_index = 2)
    discrete_logits, continuous_dist = out.discrete, out.continuous

    assert discrete_logits.shape == (2, 64, 20000)
    assert continuous_dist.shape == (2, 64, 5, 2)

    # 4. multi-discrete

    multi_discrete_inp = torch.randint(0, 500, (2, 64, 2))

    tokens = embed(multi_discrete_inp, selector_index = 3)
    assert tokens.shape == (2, 64, 512)

    logits = readout(tokens, selector_index = 3)

    assert len(logits) == 2
    assert all([logit.shape == (2, 64, 500) for logit in logits])

    sampled = readout.sample(logits, selector_index = 3)
    assert sampled.shape == (2, 64, 2)

    log_prob = readout.log_prob(logits, sampled, selector_index = 3)
    assert log_prob.shape == (2, 64, 2)

    entropy = readout.entropy(logits, selector_index = 3)
    assert entropy.shape == (2, 64, 2)

    # 5. test override return_both_discrete_and_continuous (using return_only_discrete_or_continuous = False)

    # discrete only

    tokens = embed(token_ids, selector_index = 0)
    out = readout(tokens, selector_index = 0, return_only_discrete_or_continuous = False)
    assert isinstance(out, tuple) and hasattr(out, 'discrete') and hasattr(out, 'continuous')
    assert exists(out.discrete) and not exists(out.continuous)

    # continuous only

    tokens = embed(continuous_input, selector_index = 1)
    out = readout(tokens, selector_index = 1, return_only_discrete_or_continuous = False)
    assert isinstance(out, tuple) and hasattr(out, 'discrete') and hasattr(out, 'continuous')
    assert not exists(out.discrete) and exists(out.continuous)

def test_concat_entropy_log_prob():
    # 1. mixed case

    selectors = (
        # discrete (2 groups)
        [[i for i in range(100)], [i + 100 for i in range(100)]],
        # continuous (5 dims)
        [i for i in range(5)]
    )

    embed, readout = EmbedAndReadout(
        dim = 512,
        num_discrete = 300,
        num_continuous = 30,
        selectors = selectors,
        readout_kwargs = dict(
            continuous_log_var_embed = True
        )
    )

    discrete_input = torch.randint(0, 100, (2, 64, 2)) # 2 discrete groups
    continuous_input = torch.randn(2, 64, 5)

    tokens = embed((discrete_input, continuous_input))

    logits = readout(tokens)
    sampled = readout.sample(logits)

    # test log prob concat

    log_prob = readout.log_prob(logits, sampled, concat = True)

    assert log_prob.shape == (2, 64, 2 + 5)

    # test entropy concat

    entropy = readout.entropy(logits, concat = True)

    assert entropy.shape == (2, 64, 2 + 5)

    # 2. single discrete case

    embed, readout = EmbedAndReadout(
        dim = 512,
        num_discrete = 100,
        return_only_discrete_or_continuous = True
    )

    discrete_input = torch.randint(0, 100, (2, 64))
    tokens = embed(discrete_input)
    logits = readout(tokens)
    sampled = readout.sample(logits)

    log_prob = readout.log_prob(logits, sampled, concat = True)
    assert log_prob.shape == (2, 64, 1)

    entropy = readout.entropy(logits, concat = True)
    assert entropy.shape == (2, 64, 1)

    # 3. single continuous case

    embed, readout = EmbedAndReadout(
        dim = 512,
        num_continuous = 5,
        readout_kwargs = dict(
            continuous_log_var_embed = True
        )
    )

    continuous_input = torch.randn(2, 64, 5)
    tokens = embed(continuous_input)
    dist = readout(tokens)
    sampled = readout.sample(dist)

    log_prob = readout.log_prob(dist, sampled, concat = True)
    assert log_prob.shape == (2, 64, 5)

    entropy = readout.entropy(dist, concat = True)
    assert entropy.shape == (2, 64, 5)
