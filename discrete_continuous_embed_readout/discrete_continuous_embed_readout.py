from __future__ import annotations
from collections import namedtuple
from functools import partial
from beartype import beartype

import torch
from torch import nn, Tensor, arange, tensor, is_tensor, stack, cat
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map

from torch.distributions import Normal

# einops

from einops import rearrange, reduce, repeat, einsum

# ein notation:
# nd - num discrete
# nc - num continuous
# f - feature dimension

# helpers

def exists(v):
    return v is not None

def identity(t):
    return t

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def compact(arr):
    return [*filter(exists, arr)]

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# tensor helpers

def tree_map_tensor(obj, fn):
    return tree_map(lambda t: fn(t) if is_tensor(t) else t, obj)

def exclusive_cumsum(t):
    if not is_tensor(t):
        t = tensor(t)

    t = F.pad(t, (1, 0))
    return t.cumsum(dim = -1)[..., :-1]

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# distribution related

def gumbel_noise(t, eps):
    return -log(-log(torch.rand_like(t), eps), eps)

def gumbel_sample(t, temperature = 1., eps = 1e-20):
    if temperature <= 0.:
        return t.argmax(dim = -1)

    noise = gumbel_noise(t, eps)
    t = t / max(temperature, eps) + noise
    return t.argmax(dim = -1)

def gaussian_sample(mu_log_var, temperature = 1.):
    mu, log_var = mu_log_var.unbind(dim = -1)
    std = (0.5 * log_var).exp()
    return mu + torch.rand_like(std) * temperature

# multi categorical

def gumbel_sample_multi_categorical(
    dists,
    temperature = 1.,
    eps = 1e-20
):
    assert len(dists) > 0
    one_dist = first(dists)

    nested_dists = nested_tensor(dists, layout = torch.jagged)

    if temperature > 0:
        noise = gumbel_noise(nested_dists, eps)
        nested_dists = nested_dists / max(temperature, eps) + noise

    mask_value = max_neg_value(one_dist)
    padded = nested_dists.to_padded_tensor(mask_value)
    sampled = padded.argmax(dim = -1)

    return rearrange(sampled, 'nd ... -> ... nd')

# base

class Base(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_discrete: int | tuple[int, ...] = 0,
        num_continuous: int = 0,
        continuous_log_var_embed = True,
        continuous_mean_std: Tensor | None = None,
        use_parallel_multi_discrete = True,
        eps = 1e-6
    ):
        super().__init__()
        num_discrete = cast_tuple(num_discrete) if num_discrete != 0 else ()

        total_discrete = sum(num_discrete)
        total_continuous = num_continuous * (2 if continuous_log_var_embed else 1)

        total = total_discrete + total_continuous

        assert total > 0, 'cannot have both discrete and continuous disabled'

        self.dim = dim

        # all embeddings for discrete and continuous stored together
        # order will be [discrete] [continuous]
        # discrete is further broken up by groups if tuple of ints passed in - so [discrete group 1] [discrete group 2] ... [continuous]

        self.embeddings = nn.Embedding(total, dim)
        nn.init.normal_(self.embeddings.weight, std = 1e-2)

        # continuous related

        self.has_continuous = num_continuous > 0
        self.continuous_offset = total_discrete
        self.continuous_log_var_embed = continuous_log_var_embed

        self.register_buffer('continuous_indices', arange(num_continuous) + self.continuous_offset, persistent = False)
        self.register_buffer('continuous_mean_log_var_indices', arange(total_continuous) + self.continuous_offset, persistent = False)

        # maybe norm and inverse norm

        self.can_norm_continuous = exists(continuous_mean_std)

        if self.can_norm_continuous:
            assert self.has_continuous
            assert continuous_mean_std.shape == (num_continuous, 2)
            assert (continuous_mean_std[..., -1] > 0).all()

            self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.eps = eps

        # discrete related computed values

        self.has_discrete = total_discrete > 0
        self.num_discrete = num_discrete
        self.num_discrete_groups = len(num_discrete)

        self.use_parallel_multi_discrete = use_parallel_multi_discrete # sampling, entropy, log prob in parallel for multi-discrete

        discrete_group_offsets = exclusive_cumsum(tensor(num_discrete))

        self.register_buffer('discrete_indices', arange(total_discrete), persistent = False)
        self.register_buffer('discrete_group_offsets', discrete_group_offsets, persistent = False)

        # inferring

        self.one_of_discrete_or_continuous = self.has_discrete ^ self.has_continuous

# embed and readout

class Embed(Base):
    def __init__(
        self,
        *args,
        auto_append_discrete_group_dim = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, continuous_log_var_embed = False)

        self.auto_append_discrete_group_dim = default(auto_append_discrete_group_dim, self.num_discrete_groups == 1)
        assert not (self.auto_append_discrete_group_dim and self.num_discrete_groups > 1), 'cannot have greater than one discrete group and auto-unsqueezing of a dimension'

    def forward(
        self,
        inp: Tensor | tuple[Tensor, Tensor],
        sum_discrete_groups = True,
        sum_continuous = True,
        sum_discrete_continuous = True,
        normalize_continuous = None,
        explicit_none_outputs = False
    ):
        normalize_continuous = default(normalize_continuous, self.can_norm_continuous)
        assert not (normalize_continuous and not self.can_norm_continuous)

        # handle inferring it is either discrete or continuous

        if is_tensor(inp):
            assert self.one_of_discrete_or_continuous
            dtype = inp.dtype

            if self.has_discrete:
                assert dtype in (torch.int, torch.long)
                inp = (inp, None)

            if self.has_continuous:
                assert dtype == torch.float
                inp = (None, inp)

        # destruct

        discrete, continuous = inp

        if self.auto_append_discrete_group_dim and self.has_discrete:
            discrete = rearrange(discrete, '... -> ... 1')

        # maybe norm continuous

        if self.can_norm_continuous and exists(continuous):
            mean, std = self.continuous_mean_std.unbind(dim = -1)
            continuous = (continuous - mean) / std.clamp_min(self.eps)

        # take care of discrete

        discrete_embed = None

        if exists(discrete):
            if self.num_discrete_groups > 1:
                assert discrete.shape[-1] == self.num_discrete_groups, f'shape of input must end with {self.num_discrete_groups}, as there are two discrete groups'
                discrete = discrete + self.discrete_group_offsets

            discrete_embed = self.embeddings(discrete)

            # reducing across discrete groups

            if sum_discrete_groups:
                discrete_embed = reduce(discrete_embed, '... nd d -> ... d', 'sum')

        # take care of continuous

        continuous_embed = None

        if exists(continuous):
            continuous_embed = self.embeddings(self.continuous_indices)

            # whether to reduce for continuous

            if sum_continuous:
                continuous_embed = einsum(continuous_embed, continuous, 'nc d, ... nc -> ... d')
            else:
                continuous_embed = einsum(continuous_embed, continuous, 'nc d, ... nc -> ... nc d')

        # convenience

        if self.one_of_discrete_or_continuous and not explicit_none_outputs:
            if self.has_discrete:
                return discrete_embed

            if self.has_continuous:
                return continuous_embed

        # handle if both are given

        output = (discrete_embed, continuous_embed)

        if (
            not sum_discrete_continuous or
            not sum_discrete_groups or
            not sum_continuous
        ):
            return output

        # sum into one token for transformer, often the case, but could be handled separately (say multi-stream transformer or something more elaborate)

        output = sum(compact(output))

        return output

class Readout(Base):
    def __init__(
        self,
        *args,
        return_one_discrete_logits = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.return_one_discrete_logits = default(return_one_discrete_logits, self.num_discrete_groups == 1)
        assert not (self.return_one_discrete_logits and self.num_discrete_groups > 1), 'cannot return only one discrete logit group if greater than one group'

        self.register_buffer('zero', tensor(0.), persistent = False)

    def sample_discrete(
        self,
        discrete_logits: Tensor | list[Tensor] | tuple[Tensor, ...],
        temperature = 1,
        filter_fn: Callable  = identity,
        filter_kwargs: dict = dict()
    ):
        is_list_tuple = isinstance(discrete_logits, (list, tuple))

        if not is_list_tuple:
            discrete_logits = (discrete_logits,)

        discrete_logits = [filter_fn(t, **filter_kwargs) for t in discrete_logits]

        if len(discrete_logits) > 1 and self.use_parallel_multi_discrete:
            sampled = gumbel_sample_multi_categorical(discrete_logits, temperature = temperature)
        else:
            sampled = tree_map_tensor(discrete_logits, partial(gumbel_sample, temperature = temperature))
            sampled = stack(sampled, dim = -1)

        if not is_list_tuple:
            sampled = rearrange(sampled, '... 1 -> ...')

        return sampled

    def sample_continuous(
        self,
        continuous_dist_params,
        temperature = 1.
    ):
        assert self.continuous_log_var_embed

        return gaussian_sample(continuous_dist_params, temperature)

    def sample(
        self,
        dist
    ):
        if self.one_of_discrete_or_continuous:
            if self.has_discrete:
                return self.sample_discrete(dist)

            if self.has_continuous:
                return self.sample_continuous(dist)

        discrete, continuous = dist
        return self.sample_discrete(discrete), self.sample_continuous(continuous)

    def log_prob_discrete(
        self,
        discrete_logits:  Tensor | list[Tensor] | tuple[Tensor, ...], 
        sampled,
    ):
        is_list_tuple = isinstance(discrete_logits, (list, tuple))

        if not is_list_tuple:
            need_unsqueeze = sampled.ndim == (discrete_logits.ndim - 1)

            if need_unsqueeze:
                sampled = rearrange(sampled, '... -> ... 1')

            log_prob = discrete_logits.gather(-1, sampled)

            if need_unsqueeze:
                log_prob = rearrange(log_prob, '... 1 -> ...')

            return log_prob

        assert len(discrete_logits) > 0

        lens = tensor([d.shape[-1] for d in discrete_logits], device = first(discrete_logits).device)
        offsets = exclusive_cumsum(lens)

        indices = sampled + offsets

        # handle log softmax

        if self.use_parallel_multi_discrete:
            nested = nested_tensor(discrete_logits, layout = torch.jagged)
            log_softmaxed = nested.softmax(dim = -1) * log(nested)
            log_softmaxed = log_softmaxed.unbind()
        else:
            log_softmaxed = [logit.log_softmax(dim = -1) for logit in discrete_logits]

        # gather log probs

        log_probs = cat(log_softmaxed, dim = -1).gather(-1, indices)
        return log_probs

    def log_prob_continuous(
        self,
        continuous_dist_params,
        sampled
    ):
        assert self.continuous_log_var_embed

        mean, log_var = continuous_dist_params.unbind(dim = -1)
        std = (0.5 * log_var).exp()
        return Normal(mean, std).log_prob(sampled)

    def log_prob(
        self,
        dist,
        sampled
    ):
        if self.one_of_discrete_or_continuous:
            if self.has_discrete:
                return self.log_prob_discrete(dist, sampled)

            if self.has_continuous:
                return self.log_prob_continuous(dist, sampled)

        discrete, continuous = dist
        discrete_sampled, continuous_sampled = sampled
        return self.log_prob_discrete(discrete, discrete_sampled), self.log_prob_continuous(continuous, continuous_sampled)

    def forward(
        self,
        embed,
        targets = None,
        return_loss = False,
        temperature = 1.,
        explicit_none_outputs = False
    ):
        assert xnor(exists(targets), return_loss), '`target` must be passed in if `return_loss` set to True and vice versa'

        # discrete unembedding

        discrete_logits_for_groups = None

        if self.has_discrete:
            discrete_unembed = self.embeddings(self.discrete_indices)
            all_discrete_logits = einsum(embed, discrete_unembed, '... d, nd d -> ... nd')

            discrete_logits_for_groups = all_discrete_logits.split(self.num_discrete, dim = -1)

        # continuous unembedding

        continuous_dist_params = None

        if self.has_continuous:
            continuous_unembed = self.embeddings(self.continuous_mean_log_var_indices)
            continuous_dist_params = einsum(embed, continuous_unembed, '... d, nc d -> ... nc')

            if self.continuous_log_var_embed:
                continuous_dist_params = rearrange(continuous_dist_params, '... (mu_logvar nc) -> ... nc mu_logvar', mu_logvar = 2)

        # maybe only return distribution parameters

        if not return_loss:
            if self.return_one_discrete_logits and exists(discrete_logits_for_groups):
                discrete_logits_for_groups = first(discrete_logits_for_groups)

            if self.one_of_discrete_or_continuous and not explicit_none_outputs:
                if self.has_discrete:
                    return discrete_logits_for_groups

                if self.has_continuous:
                    return continuous_dist_params

            return discrete_logits_for_groups, continuous_dist_params

        # handle destructing of target

        discrete_targets = targets
        continuous_targets = targets

        if self.has_discrete and self.has_continuous:
            assert isinstance(targets, (tuple, list)) and len(targets) == 2
            discrete_targets, continuous_targets = targets

        # take care of only one discrete logit group, as in language modeling

        if self.return_one_discrete_logits:
            discrete_targets = rearrange(discrete_targets, '... -> ... 1')

        # handle basic losses

        discrete_losses = self.zero

        if self.has_discrete:
            discrete_losses = tuple(F.cross_entropy(rearrange(discrete_logit, 'b ... nd -> b nd ...'), one_target) for discrete_logit, one_target in zip(discrete_logits_for_groups, discrete_targets.unbind(dim = -1)))

            discrete_losses = sum(discrete_losses)

        continuous_losses = self.zero

        if self.has_continuous:

            if self.continuous_log_var_embed:
                mean, log_var = continuous_dist_params.unbind(dim = -1)
                std = (0.5 * log_var).exp()

                gaussian = Normal(mean, std)

                continuous_losses = -gaussian.log_prob(continuous_targets)
            else:
                continuous_losses = F.mse_loss(continuous_dist_params, continuous_targets, reduction = 'none')

            continuous_losses = reduce(continuous_losses, '... nc -> ...', 'sum')

            continuous_losses = continuous_losses.mean()

        return discrete_losses + continuous_losses

# helper functions for creating both, with optional weight tying

def EmbedAndReadout(
    *args,
    weight_tie = False,
    embed_kwargs: dict = dict(),
    readout_kwargs: dict = dict(),
    **kwargs,
):
    embed = Embed(*args, **embed_kwargs, **kwargs)
    readout = Readout(*args, **readout_kwargs, **kwargs)

    if weight_tie:
        embed.embeddings = readout.embeddings # readout has the superset

    return embed, readout
