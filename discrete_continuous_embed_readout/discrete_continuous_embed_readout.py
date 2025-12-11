from __future__ import annotations
from collections import namedtuple
from beartype import beartype

import torch
from torch import nn, Tensor, arange, tensor, is_tensor
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.nn import Module, ModuleList

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

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def exclusive_cumsum(t):
    if not is_tensor(t):
        t = tensor(t)

    t = F.pad(t, (1, 0))
    return t.cumsum(dim = -1)[..., :-1]

# base

class Base(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_discrete: int | tuple[int, ...] = 0,
        num_continuous: int = 0,
        continuous_log_var_embed = True
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

        # discrete related computed values

        self.has_discrete = total_discrete > 0
        self.num_discrete = num_discrete
        self.num_discrete_groups = len(num_discrete)

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
    ):
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

        # take care of discrete

        discrete_embed = None

        if exists(discrete):
            if self.num_discrete_groups > 1:
                assert discrete.shape[-1] == self.num_discrete_groups
                discrete = discrete + self.discrete_group_offsets

            discrete_embed = self.embeddings(discrete)

            # reducing across discrete groups

            if sum_continuous:
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

        if self.one_of_discrete_or_continuous:
            if self.has_discrete:
                return discrete_embed

            if self.has_continuous:
                return continuous_embed

        # handle if both are given

        output = (discrete_embed, continuous_embed)

        if not sum_discrete_continuous:
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

    def forward(
        self,
        embed,
        targets = None,
        return_loss = False
    ):
        assert xnor(exists(targets), return_loss), '`target` must be passed in if `return_loss` set to True and vice versa'

        discrete_logits = None

        # discrete unembedding

        if self.has_discrete:
            discrete_unembed = self.embeddings(self.discrete_indices)
            all_discrete_logits = einsum(embed, discrete_unembed, '... d, nd d -> ... nd')

            discrete_logits_for_groups = all_discrete_logits.split(self.num_discrete, dim = -1)

        # continuous unembedding

        if self.has_continuous:
            continuous_unembed = self.embeddings(self.continuous_mean_log_var_indices)
            continous_dist_params = einsum(embed, continuous_unembed, '... d, nc d -> ... nc')

            if self.continuous_log_var_embed:
                continous_dist_params = rearrange(continous_dist_params, '... (mu_logvar nc) -> ... nc mu_logvar', mu_logvar = 2)

        # maybe only return distribution parameters

        if not return_loss:
            if self.return_one_discrete_logits and exists(discrete_logits_for_groups):
                discrete_logits_for_groups = first(discrete_logits_for_groups)

            if self.one_of_discrete_or_continuous:
                if self.has_discrete:
                    return discrete_logits_for_groups

                if self.has_continuous:
                    return continous_dist_params

            return discrete_logits_for_groups, continous_dist_params

        # handle destructing of target

        discrete_targets = targets
        continuous_targets = targets

        if self.has_discrete and self.has_continuous:
            assert isinstance(targets, (tuple, list)) and len(targets) == 2
            discrete_targets, continuous_targets = targets

        # take care of only one discrete logit group, as in language modeling

        if self.return_one_discrete_logits:
            discrete_targets = cast_tuple(discrete_targets)

        # handle basic losses

        discrete_losses = self.zero

        if self.has_discrete:
            discrete_losses = tuple(F.cross_entropy(rearrange(discrete_logit, 'b ... nd -> b nd ...'), one_target) for discrete_logit, one_target in zip(discrete_logits_for_groups, discrete_targets))

            discrete_losses = sum(discrete_losses)

        continuous_losses = self.zero

        if self.has_continuous:

            if self.continuous_log_var_embed:
                mean, log_var = continous_dist_params.unbind(dim = -1)
                std = (0.5 * log_var).exp()

                gaussian = Normal(mean, std)

                continuous_losses = -gaussian.log_prob(continuous_targets)
            else:
                continuous_losses = F.mse_loss(continous_dist_params, continuous_targets, reduction = 'none')

            continuous_losses = reduce(continuous_losses, '... nc -> ...', 'sum')

            continuous_losses = continuous_losses.mean()

        return discrete_losses + continuous_losses
