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
    MultiCategorical,
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

    selector = (
        # discrete (2 groups)
        [[i for i in range(100)], [i + 100 for i in range(100)]],
        # continuous (5 dims)
        [i for i in range(5)]
    )

    embed, readout = EmbedAndReadout(
        dim = 512,
        num_discrete = 300,
        num_continuous = 30,
        selector = selector,
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

def test_concat_discrete_continuous():
    embed, _ = EmbedAndReadout(512, num_discrete = 20_000, num_continuous = 5)

    # 1. mixed case

    discrete_input = torch.randint(0, 20_000, (2, 64))
    continuous_input = torch.randn(2, 64, 5)

    tokens = embed((discrete_input, continuous_input), concat_discrete_continuous = True)
    assert tokens.shape == (2, 64, 2, 512)

    # 2. discrete only

    tokens = embed((discrete_input, None), selector_index = 0, concat_discrete_continuous = True)
    assert tokens.shape == (2, 64, 1, 512)

    # 3. continuous only

    tokens = embed((None, continuous_input), selector_index = 0, concat_discrete_continuous = True)
    assert tokens.shape == (2, 64, 1, 512)

    # 4. mixed case with no sum

    embed = Embed(
        512,
        num_discrete = (1000, 1000),
        num_continuous = 5
    )

    discrete_input = torch.randint(0, 1000, (2, 64, 2))
    continuous_input = torch.randn(2, 64, 5)

    tokens = embed(
        (discrete_input, continuous_input),
        sum_discrete_sets = False,
        sum_continuous = False,
        concat_discrete_continuous = True
    )

    assert tokens.shape == (2, 64, 2 + 5, 512)

    # 5. mixed case with sum continuous but not discrete

    tokens = embed(
        (discrete_input, continuous_input),
        sum_discrete_sets = False,
        sum_continuous = True,
        concat_discrete_continuous = True
    )

    assert tokens.shape == (2, 64, 2 + 1, 512)


def test_auto_squeeze_single_output():

    embed, readout = EmbedAndReadout(
        dim = 512,
        num_discrete = 100,
        return_only_discrete_or_continuous = True, # to isolate discrete
        readout_kwargs = dict(
            auto_squeeze_single_output = False
        )
    )

    embed_squeezed, readout_squeezed = EmbedAndReadout(
        dim = 512,
        num_discrete = 100,
        return_only_discrete_or_continuous = True,
        readout_kwargs = dict(
            auto_squeeze_single_output = True
        )
    )

    discrete_input = torch.randint(0, 100, (2, 64))

    tokens = embed(discrete_input)

    logits_unsqueezed = readout(tokens)

    assert isinstance(logits_unsqueezed, (list, tuple))
    assert len(logits_unsqueezed) == 1
    assert logits_unsqueezed[0].shape == (2, 64, 100)

    sampled_unsqueezed = readout.sample(logits_unsqueezed)

    assert sampled_unsqueezed.shape == (2, 64, 1)

    log_probs_unsqueezed = readout.log_prob(logits_unsqueezed, sampled_unsqueezed)

    assert log_probs_unsqueezed.shape == (2, 64, 1)

    entropy_unsqueezed = readout.entropy(logits_unsqueezed)

    assert entropy_unsqueezed.shape == (2, 64, 1)

    logits_squeezed = readout_squeezed(tokens)
    assert isinstance(logits_squeezed, torch.Tensor)
    assert logits_squeezed.shape == (2, 64, 100)

    sampled_squeezed = readout_squeezed.sample(logits_squeezed)
    assert sampled_squeezed.shape == (2, 64)

    log_probs_squeezed = readout_squeezed.log_prob(logits_squeezed, sampled_squeezed)
    assert log_probs_squeezed.shape == (2, 64)

    entropy_squeezed = readout_squeezed.entropy(logits_squeezed)
    assert entropy_squeezed.shape == (2, 64)

    embed_continuous, readout_continuous = EmbedAndReadout(
        dim = 512,
        num_continuous = 1, # single continuous dim
        readout_kwargs = dict(
            auto_squeeze_single_output = False,
            continuous_log_var_embed = True
        )
    )

    continuous_input = torch.randn(2, 64, 1)
    tokens_continuous = embed_continuous(continuous_input)
    dist_continuous = readout_continuous(tokens_continuous) # (B, T, 1, 2)
    assert dist_continuous.shape == (2, 64, 1, 2)

    sampled_continuous = readout_continuous.sample(dist_continuous)
    assert sampled_continuous.shape == (2, 64, 1)

    embed_continuous_sq, readout_continuous_sq = EmbedAndReadout(
        dim = 512,
        num_continuous = 1,
        readout_kwargs = dict(
            auto_squeeze_single_output = True,
            continuous_log_var_embed = True
        )
    )

    dist_continuous_sq = readout_continuous_sq(tokens_continuous)
    assert dist_continuous_sq.shape == (2, 64, 1, 2)

def test_readout_unreduced_loss():
    embed, readout = EmbedAndReadout(512, num_discrete = 100, num_continuous = 5)

    discrete_input = torch.randint(0, 100, (2, 64))
    continuous_input = torch.randn(2, 64, 5)

    tokens = embed((discrete_input, continuous_input))

    # test unreduced loss

    discrete_loss, continuous_loss = readout(
        tokens,
        (discrete_input, continuous_input),
        return_loss = True,
        return_unreduced_loss = True
    )

    assert discrete_loss.shape == (2, 64, 1)
    assert continuous_loss.shape == (2, 64, 5)

    # test masked loss

    mask = torch.randint(0, 2, (2, 64)).bool()

    masked_loss = readout(
        tokens,
        (discrete_input, continuous_input),
        return_loss = True,
        loss_mask = mask
    )

    discrete_loss_masked = discrete_loss * mask[..., None]
    continuous_loss_masked = continuous_loss * mask[..., None]

    expected_discrete_loss = discrete_loss_masked.sum() / mask.sum()
    expected_continuous_loss = continuous_loss_masked.sum() / mask.sum()

    assert torch.allclose(masked_loss.discrete, expected_discrete_loss)
    assert torch.allclose(masked_loss.continuous, expected_continuous_loss)

    # test unreduced masked loss

    unreduced_masked_loss = readout(
        tokens,
        (discrete_input, continuous_input),
        return_loss = True,
        return_unreduced_loss = True,
        loss_mask = mask
    )

    assert torch.allclose(unreduced_masked_loss.discrete, discrete_loss_masked)
    assert torch.allclose(unreduced_masked_loss.continuous, continuous_loss_masked)

def test_runtime_selector_config():
    # 3 discrete, 1 continuous
    dim = 32
    num_discrete = 10
    num_continuous = 2

    embed = Embed(
        dim = dim,
        num_discrete = num_discrete,
        num_continuous = num_continuous
    )

    readout = Readout(
        dim = dim,
        num_discrete = num_discrete,
        num_continuous = num_continuous
    )

    discrete_config = [[0, 1, 2, 3, 4]]
    continuous_config = [0]
    selector_config = (discrete_config, continuous_config)

    batch_size = 4

    # Test Embed with runtime config

    discrete_inp = torch.randint(0, 5, (batch_size,))
    continuous_inp = torch.randn(batch_size, 1)

    embedded = embed(
        (discrete_inp, continuous_inp),
        selector_config = selector_config
    )

    assert embedded.shape == (batch_size, dim)

    logits = readout(
        embedded,
        selector_config = selector_config,
        return_only_discrete_or_continuous = False
    )

    discrete_logits, continuous_params = logits

    assert torch.is_tensor(discrete_logits)
    assert discrete_logits.shape == (batch_size, 5)

    assert continuous_params.shape == (batch_size, 1, 2)

    sampled = readout.sample(logits, selector_config = selector_config)

    log_probs = readout.log_prob(logits, sampled, selector_config = selector_config)

    assert isinstance(log_probs, tuple) # DiscreteContinuous
    assert log_probs.discrete.shape == (batch_size,)
    assert log_probs.continuous.shape == (batch_size, 1)

    # Test entropy
    entropy = readout.entropy(logits, selector_config = selector_config)

    # Test calculate_loss

    discrete_targets = torch.randint(0, 5, (batch_size,))
    continuous_targets = torch.randn(batch_size, 1)


    loss = readout.calculate_loss(
        logits,
        (discrete_targets, continuous_targets),
        selector_config = selector_config
    )

    assert isinstance(loss, tuple) # DiscreteContinuous

def test_lone_discrete_config():
    dim = 32
    num_discrete = 10
    num_continuous = 2

    embed = Embed(
        dim = dim,
        num_discrete = num_discrete,
        num_continuous = num_continuous
    )

    readout = Readout(
        dim = dim,
        num_discrete = num_discrete,
        num_continuous = num_continuous
    )

    # Test passing lone discrete config as list[list[int]]
    discrete_config = [[0, 1, 2]]

    batch_size = 4
    discrete_inp = torch.randint(0, 3, (batch_size,))

    # Embed
    embedded = embed(
        discrete_inp,
        selector_config = discrete_config
    )
    assert embedded.shape == (batch_size, dim)

    # Readout
    logits = readout(
        embedded,
        selector_config = discrete_config
    )

    assert torch.is_tensor(logits)
    assert logits.shape == (batch_size, 3)

    sampled = readout.sample(logits, selector_config = discrete_config)
    assert sampled.shape == (batch_size,)

def test_lone_continuous_config():
    dim = 32
    num_discrete = 10
    num_continuous = 2

    embed = Embed(
        dim = dim,
        num_discrete = num_discrete,
        num_continuous = num_continuous
    )

    readout = Readout(
        dim = dim,
        num_discrete = num_discrete,
        num_continuous = num_continuous,
        continuous_log_var_embed = True
    )

    continuous_config = [0, 1]

    batch_size = 4
    continuous_inp = torch.randn(batch_size, 2)

    embedded = embed(
        continuous_inp,
        selector_config = continuous_config
    )
    assert embedded.shape == (batch_size, dim)

    dist = readout(
        embedded,
        selector_config = continuous_config
    )

    assert torch.is_tensor(dist)
    assert dist.shape == (batch_size, 2, 2)


    sampled = readout.sample(dist, selector_config = continuous_config)
    assert sampled.shape == (batch_size, 2)

from discrete_continuous_embed_readout.discrete_continuous_embed_readout import ParameterlessReadout, rearrange

def test_parameterless_readout():
    num_discrete = 10
    num_continuous = 2

    readout = ParameterlessReadout(
        num_discrete = num_discrete,
        num_continuous = num_continuous,
        continuous_log_var_embed = True
    )

    batch_size = 4

    # create dummy logits
    discrete_logits = torch.randn(batch_size, num_discrete)
    continuous_params = torch.randn(batch_size, num_continuous * 2)
    # continuous_params need to be shaped (batch, num_continuous, 2)
    continuous_params = rearrange(continuous_params, '... (nc d) -> ... nc d', d = 2)

    logits = (discrete_logits, continuous_params)

    # 1. Test sample
    sampled = readout.sample(logits)
    discrete_sampled, continuous_sampled = sampled

    assert discrete_sampled.shape == (batch_size,)
    assert continuous_sampled.shape == (batch_size, num_continuous)

    # 2. Test log_prob
    log_probs = readout.log_prob(logits, sampled)
    assert log_probs.discrete.shape == (batch_size,)
    assert log_probs.continuous.shape == (batch_size, num_continuous)

    # 3. Test entropy
    entropies = readout.entropy(logits)
    assert entropies.discrete.shape == (batch_size,)
    assert entropies.continuous.shape == (batch_size, num_continuous)

    # 4. Test calculate_loss
    discrete_targets = torch.randint(0, num_discrete, (batch_size,))
    continuous_targets = torch.randn(batch_size, num_continuous)

    loss = readout.calculate_loss(logits, (discrete_targets, continuous_targets))
    assert loss.discrete.numel() == 1 # scalar loss
    assert loss.continuous.numel() == 1 # scalar loss

    # 5. Test forward raises error
    # forward attempts to project from embeddings which don't exist
    try:
        readout(torch.randn(batch_size, 512))
        assert False, 'should have raised error'
    except RuntimeError as e:
        assert 'embeddings not present' in str(e) or 'object has no attribute' in str(e)

def test_explicit_single_action_dim_given():
    embed, readout = EmbedAndReadout(
        dim = 512,
        num_discrete = 100,
        explicit_single_action_dim_given = True
    )

    discrete_input = torch.randint(0, 100, (2, 64, 1)) # explicitly 1 action dim

    tokens = embed(discrete_input)
    assert tokens.shape == (2, 64, 512)

    logits = readout(tokens)
    assert isinstance(logits, (list, tuple)) # auto_squeeze_single_output is False
    assert len(logits) == 1
    assert logits[0].shape == (2, 64, 100)

    sampled = readout.sample(logits)
    assert sampled.shape == (2, 64, 1) # no auto squeeze

    log_probs = readout.log_prob(logits, sampled)
    assert log_probs.shape == (2, 64, 1)

    entropy = readout.entropy(logits)
    assert entropy.shape == (2, 64, 1)
