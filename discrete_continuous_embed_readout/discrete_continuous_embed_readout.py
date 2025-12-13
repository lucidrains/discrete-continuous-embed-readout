from __future__ import annotations
from typing import Callable
from beartype import beartype
from beartype.door import is_bearable

from collections import namedtuple
from functools import partial
from itertools import count

import torch
from torch import nn, Tensor, arange, tensor, is_tensor, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map

from torch.distributions import Normal

# einops

from einops import rearrange, reduce, repeat, einsum

# ein notation:
# nd - num discrete
# nc - num continuous
# f - feature dimension
# l - logits

# constants

DiscreteContinuous = namedtuple('DiscreteContinuous', ('discrete', 'continuous'))

DiscreteConfig = list[list[int]]
ContinuousConfig = list[int]
SelectorConfig = tuple[DiscreteConfig, ContinuousConfig] | DiscreteConfig | ContinuousConfig

# helpers

def exists(v):
    return v is not None

def identity(t):
    return t

def first(arr):
    return arr[0]

def xnor(x, y):
    return x == y

def compact(arr):
    return [*filter(exists, arr)]

def default(v, d):
    return v if exists(v) else d

def flatten(arr):
    return [el for sub_arr in arr for el in sub_arr]

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

def safe_cat(tensors, dim = 0):
    tensors = [*filter(exists, tensors)]
    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return cat(tensors, dim = dim)

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

def calc_entropy(t, eps = 1e-20):
    prob = t.softmax(dim = -1)
    return (-prob * log(prob, eps)).sum(dim = -1)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cat_with_lens(tensors: list[Tensor]):
    catted_tensors = cat(tensors, dim = -1)
    lens = tensor([t.shape[-1] for t in tensors], device = catted_tensors.device)
    return catted_tensors, lens

def segmented_softmax(flat_logits, lengths):
    if isinstance(lengths, (tuple, list)):
        lengths = tensor(lengths, device = flat_logits.device)

    flat_logits = rearrange(flat_logits, '... d -> d ...')

    # max for stability

    max_logits = torch.segment_reduce(flat_logits, 'max', lengths = lengths)
    max_logits = torch.repeat_interleave(max_logits, lengths, dim = 0)

    flat_logits = flat_logits - max_logits.detach()

    # exponentiate

    exp_logits = flat_logits.exp()

    # divisor

    sum_exp = torch.segment_reduce(exp_logits, 'sum', lengths = lengths)
    sum_exp = torch.repeat_interleave(sum_exp, lengths, dim = 0)

    output = exp_logits / sum_exp

    output = rearrange(output, 'd ... -> ... d')
    return output

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
    return mu + torch.randn_like(std) * temperature

def mean_log_var_to_normal_dist(mean_log_var):
    mean, log_var = mean_log_var.unbind(dim = -1)
    std = (0.5 * log_var).exp()
    return Normal(mean, std)

# multi categorical

def gumbel_sample_multi_categorical(
    dists,
    temperature = 1.,
    eps = 1e-20
):
    is_greedy = temperature <= 0.
    assert len(dists) > 0
    one_dist = first(dists)

    dists, lens = cat_with_lens(dists)

    if not is_greedy:
        noise = gumbel_noise(dists, eps)
        dists = dists / max(temperature, eps) + noise

    dists = dists.split(lens.tolist(), dim = -1)
    max_len = max(lens.tolist())

    mask_value = max_neg_value(one_dist)
    padded = [F.pad(d, (0, max_len - d.shape[-1]), value = mask_value) for d in dists]

    sampled = stack(padded, dim = -2).argmax(dim = -1)

    return sampled

# buffer container, so continuous selector can contain a reference

class BufferModule(Module):
    def __init__(self, tensor):
        super().__init__()
        self.register_buffer('data', tensor)

# selectors

class DiscreteSelector(Module):
    @beartype
    def __init__(
        self,
        discrete_set_indices: DiscreteConfig,
        embeddings: nn.Embedding,
    ):
        super().__init__()

        discrete_set_lens = list(map(len, discrete_set_indices))
        discrete_set_offsets = exclusive_cumsum(discrete_set_lens)

        self.num_discrete_sets = len(discrete_set_indices)

        self.embeddings = embeddings

        self.register_buffer('discrete_set_lens', tensor(discrete_set_lens), persistent = False)
        self.register_buffer('discrete_indices', tensor(flatten(discrete_set_indices)), persistent = False)
        self.register_buffer('discrete_set_offsets', discrete_set_offsets, persistent = False)

    def embed(
        self,
        indices
    ):
        if self.num_discrete_sets > 1:
            assert indices.shape[-1] == self.num_discrete_sets, f'shape of input must end with {self.num_discrete_sets}, as there are two discrete groups'
            indices = indices + self.discrete_set_offsets

        embed_indices = self.discrete_indices[indices]

        return self.embeddings(embed_indices)

    def get_readout_embeds(
        self
    ):
        return self.embeddings(self.discrete_indices)

    def split_packed(
        self,
        logits
    ):
        return logits.split(self.discrete_set_lens.tolist(), dim = -1)

class ContinuousSelector(Module):
    @beartype
    def __init__(
        self,
        continuous_indices: ContinuousConfig,
        embed: nn.Embedding,
        num_continuous,
        embedding_offset,
        continuous_mean_std: Module | None,
        continuous_log_var_embed
    ):
        super().__init__()
        # embedding is [discrete] [continuous mean] [?continuous log var]

        continuous_indices = tensor(continuous_indices)
        assert continuous_indices.unique().numel() == continuous_indices.numel()

        continuous_log_var_indices = None
        if continuous_log_var_embed:
            continuous_log_var_indices = continuous_indices + num_continuous

        self.embed = embed
        self.continuous_mean_std = continuous_mean_std
        self.continuous_log_var_embed = continuous_log_var_embed

        self.register_buffer('continuous_indices', continuous_indices + embedding_offset, persistent = False)
        self.register_buffer('continuous_mean_log_var_indices', safe_cat((continuous_indices, continuous_log_var_indices)), persistent = False)

    def get_embed(self):
        return self.embed(self.continuous_indices)

    def get_mean_logvar_embed(self):
        return self.embed(self.continuous_mean_log_var_indices)

# base

class DiscreteContinuousSelector(Module):
    @beartype
    def __init__(
        self,
        continuous_log_var_embed = True,
        continuous_mean_std: Module | None = None,
        embeddings: nn.Embedding | None = None,
        # discrete specific
        discrete_set_indices: list[list[int]] | None = None,
        # continuous specific
        continuous_indices: list[int] | None = None,
        num_continuous: int = 0,
        embedding_offset: int = 0,
    ):
        super().__init__()

        # determine if has discrete or continuous

        self.has_discrete = exists(discrete_set_indices)
        self.has_continuous = exists(continuous_indices)

        assert self.has_discrete or self.has_continuous, 'must have either discrete or continuous'

        # inferring

        self.one_of_discrete_or_continuous = self.has_discrete ^ self.has_continuous

        # discrete

        self.discrete_selector = None

        if self.has_discrete:
            self.discrete_selector = DiscreteSelector(
                discrete_set_indices,
                embeddings
            )

        # continuous

        self.continuous_selector = None

        if self.has_continuous:
            self.continuous_selector = ContinuousSelector(
                continuous_indices,
                embeddings,
                num_continuous = num_continuous,
                embedding_offset = embedding_offset,
                continuous_mean_std = continuous_mean_std,
                continuous_log_var_embed = continuous_log_var_embed
            )

    @property
    def discrete_indices(self):
        return self.discrete_selector.discrete_indices

    @property
    def continuous_mean_std(self):
        return self.continuous_selector.continuous_mean_std

    @property
    def continuous_log_var_embed(self):
        return self.continuous_selector.continuous_log_var_embed

    # methods for inferring whether to return tuple or single value

    def validate_and_return_inputs(
        self,
        inp
    ):
        if is_tensor(inp):
            assert self.one_of_discrete_or_continuous
            dtype = inp.dtype

            if dtype in (torch.int, torch.long) and self.has_discrete:
                inp = (inp, None)
            elif dtype == torch.float and self.has_continuous:
                inp = (None, inp)
            else:
                raise ValueError('invalid tensor')

        return inp


# base

class Base(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_discrete: int | tuple[int, ...] = 0,
        num_continuous: int = 0,
        selector: SelectorConfig | None = None,
        selectors: list[SelectorConfig] | None = None,
        continuous_log_var_embed = True,
        continuous_mean_std: Tensor | None = None,
        use_parallel_multi_discrete = True,
        return_only_discrete_or_continuous = True,
        eps = 1e-6
    ):
        super().__init__()

        # automatically handle single selector being passed in

        assert not (exists(selector) and exists(selectors)), 'you can only pass in `selector` or `selectors`, not both'

        if exists(selector):
            selectors = [selector]

        has_selectors = exists(selectors)

        if has_selectors:
            assert isinstance(num_discrete, int), 'num_discrete must be an int (total size of discrete embedding) if selectors are provided'

        num_discrete = cast_tuple(num_discrete) if num_discrete != 0 else ()

        total_discrete = sum(num_discrete)
        total_continuous = num_continuous * (2 if continuous_log_var_embed else 1)

        total = total_discrete + total_continuous

        # validate that num_discrete and num_continuous encompasses the max indices in selectors

        if has_selectors:
            max_discrete_index = -1
            max_continuous_index = -1

            assert len(selectors) > 0

            # normalize selectors

            selectors_configs = []

            for selector in selectors:

                discrete_indices = None
                continuous_indices = None

                if is_bearable(selector, tuple[DiscreteConfig, ContinuousConfig]):
                    discrete_indices, continuous_indices = selector
                elif is_bearable(selector, DiscreteConfig):
                    discrete_indices = selector
                elif is_bearable(selector, ContinuousConfig):
                    continuous_indices = selector
                else:
                    raise ValueError(f'invalid selector config {selector}')

                if exists(discrete_indices):
                    for group in discrete_indices:
                        if len(group) > 0:
                            max_discrete_index = max(max_discrete_index, max(group))

                if exists(continuous_indices):
                    if len(continuous_indices) > 0:
                        max_continuous_index = max(max_continuous_index, max(continuous_indices))

                selectors_configs.append((discrete_indices, continuous_indices))

            if max_discrete_index >= 0:
                assert max_discrete_index < total_discrete

            if max_continuous_index >= 0:
                assert max_continuous_index < num_continuous

        # infer has discrete or continuous

        self.has_discrete = total_discrete > 0
        self.has_continuous = num_continuous > 0

        assert total > 0, 'cannot have both discrete and continuous disabled'

        self.dim = dim

        # all embeddings for discrete and continuous stored together
        # order will be [discrete] [continuous]
        # discrete is further broken up by groups if tuple of ints passed in - so [discrete group 1] [discrete group 2] ... [continuous]

        self.embeddings = nn.Embedding(total, dim)
        nn.init.normal_(self.embeddings.weight, std = 1e-2)

        # maybe norm and inverse norm

        self.can_norm_continuous = exists(continuous_mean_std)

        if self.can_norm_continuous:
            assert self.has_continuous
            assert continuous_mean_std.shape == (num_continuous, 2)
            assert (continuous_mean_std[..., -1] > 0).all()

            continuous_mean_std = BufferModule(continuous_mean_std)

        # discrete related computed values

        self.use_parallel_multi_discrete = use_parallel_multi_discrete # sampling, entropy, log prob in parallel for multi-discrete

        # handle selectors

        self.selectors = ModuleList([])

        if not has_selectors:
            counter = count(0)
            default_discrete_indices = [[next(counter) for _ in range(n)] for n in num_discrete] if self.has_discrete else None
            default_continuous_indices = arange(num_continuous).tolist() if self.has_continuous else None

            selectors_configs = [(default_discrete_indices, default_continuous_indices)]

        for discrete_indices, continuous_indices in selectors_configs:
            self.selectors.append(DiscreteContinuousSelector(
                continuous_log_var_embed = continuous_log_var_embed,
                continuous_mean_std = continuous_mean_std,
                embeddings = self.embeddings,
                discrete_set_indices = discrete_indices,
                continuous_indices = continuous_indices,
                num_continuous = num_continuous,
                embedding_offset = total_discrete
            ))

        self.num_discrete_sets = len(num_discrete)

        # delegation properties

        self.return_only_discrete_or_continuous = return_only_discrete_or_continuous

        # epsilon

        self.eps = eps

    def get_selector(self, selector_index = None):
        if len(self.selectors) == 1:
            return self.selectors[0]

        assert exists(selector_index)
        return self.selectors[selector_index]

# embed and readout

class Embed(Base):
    def __init__(
        self,
        *args,
        auto_append_discrete_set_dim = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, continuous_log_var_embed = False)

        self.auto_append_discrete_set_dim = default(auto_append_discrete_set_dim, self.num_discrete_sets == 1)
        assert not (self.auto_append_discrete_set_dim and self.num_discrete_sets > 1), 'cannot have greater than one discrete group and auto-unsqueezing of a dimension'

    def forward(
        self,
        inp: Tensor | tuple[Tensor, Tensor],
        sum_discrete_sets = True,
        sum_continuous = True,
        sum_discrete_continuous = True,
        normalize_continuous = None,
        return_only_discrete_or_continuous = None,
        selector_index = None,
        concat_discrete_continuous = False
    ):
        normalize_continuous = default(normalize_continuous, self.can_norm_continuous)
        return_only_discrete_or_continuous = default(return_only_discrete_or_continuous, self.return_only_discrete_or_continuous)

        assert not (normalize_continuous and not self.can_norm_continuous)

        selector = self.get_selector(selector_index)

        # handle inferring it is either discrete or continuous

        inp = selector.validate_and_return_inputs(inp)

        # destruct

        discrete, continuous = inp

        if (
            exists(discrete) and
            selector.has_discrete and
            selector.discrete_selector.num_discrete_sets == 1 and
            self.auto_append_discrete_set_dim
        ):
            discrete = rearrange(discrete, '... -> ... 1')

        # maybe norm continuous

        if self.can_norm_continuous and exists(continuous):
            mean, std = selector.continuous_mean_std.data.unbind(dim = -1)
            continuous = (continuous - mean) / std.clamp_min(self.eps)

        # take care of discrete

        discrete_embed = None

        if exists(discrete) and selector.has_discrete:
            discrete_embed = selector.discrete_selector.embed(discrete)

            # reducing across discrete groups

            if sum_discrete_sets:
                discrete_embed = reduce(discrete_embed, '... nd d -> ... d', 'sum')

        # take care of continuous

        continuous_embed = None

        if exists(continuous) and selector.has_continuous:
            continuous_embed = selector.continuous_selector.get_embed()

            # whether to reduce for continuous

            if sum_continuous:
                continuous_embed = einsum(continuous_embed, continuous, 'nc d, ... nc -> ... d')
            else:
                continuous_embed = einsum(continuous_embed, continuous, 'nc d, ... nc -> ... nc d')

        # convenience

        if concat_discrete_continuous:
            ret = []
            if exists(discrete_embed):
                ret.append(rearrange(discrete_embed, '... d -> ... 1 d') if sum_discrete_sets else discrete_embed)

            if exists(continuous_embed):
                ret.append(rearrange(continuous_embed, '... d -> ... 1 d') if sum_continuous else continuous_embed)

            return cat(ret, dim = -2)

        if selector.one_of_discrete_or_continuous and return_only_discrete_or_continuous:
            if selector.has_discrete:
                return discrete_embed

            if selector.has_continuous:
                return continuous_embed

        # handle if both are given

        output = DiscreteContinuous(discrete_embed, continuous_embed)

        if (
            not sum_discrete_continuous or
            not sum_discrete_sets or
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
        self.return_one_discrete_logits = default(return_one_discrete_logits, self.num_discrete_sets == 1)
        assert not (self.return_one_discrete_logits and self.num_discrete_sets > 1), 'cannot return only one discrete logit group if greater than one group'

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
        temperature = 1.,
        selector = None
    ):
        assert exists(selector)
        assert selector.continuous_log_var_embed

        sampled = gaussian_sample(continuous_dist_params, temperature)

        if not self.can_norm_continuous:
            return sampled

        mean, std = selector.continuous_mean_std.data.unbind(dim = -1)
        inverse_normed = sampled * std + mean
        return inverse_normed

    def sample(
        self,
        dist,
        selector_index = None
    ):
        selector = self.get_selector(selector_index)

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                return self.sample_discrete(dist)

            if selector.has_continuous:
                return self.sample_continuous(dist, selector = selector)

        discrete, continuous = dist
        return self.sample_discrete(discrete), self.sample_continuous(continuous, selector = selector)

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
            discrete_logits, lens = cat_with_lens(discrete_logits)
            log_softmaxed = log(segmented_softmax(discrete_logits, lens))
        else:
            log_softmaxed = [logit.log_softmax(dim = -1) for logit in discrete_logits]
            log_softmaxed = cat(log_softmaxed, dim = -1)

        # gather log probs

        log_probs = log_softmaxed.gather(-1, indices)
        return log_probs

    def log_prob_continuous(
        self,
        continuous_dist_params,
        sampled,
        selector = None
    ):
        assert exists(selector)

        assert selector.continuous_log_var_embed
        dist = mean_log_var_to_normal_dist(continuous_dist_params)
        return dist.log_prob(sampled)

    def maybe_concat(self, output, concat = False):
        if not concat:
            return output

        if isinstance(output, DiscreteContinuous):
            output = (output.discrete, output.continuous)

        output = cast_tuple(output)
        output = [t for t in output if exists(t)]

        if len(output) == 0:
            return None

        # if any tensor is (batch, seq) - assume it is single discrete and unsqueeze (so it becomes (batch, seq, 1))

        output = [rearrange(t, '... -> ... 1') if (t.ndim == 2 and self.return_one_discrete_logits) else t for t in output]

        return cat(output, dim = -1)

    def log_prob(
        self,
        dist,
        sampled,
        selector_index = None,
        concat = False
    ):
        selector = self.get_selector(selector_index)

        output = None

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                output = self.log_prob_discrete(dist, sampled)

            elif selector.has_continuous:
                output = self.log_prob_continuous(dist, sampled, selector = selector)

        else:
            discrete, continuous = dist
            discrete_sampled, continuous_sampled = sampled
            output = DiscreteContinuous(self.log_prob_discrete(discrete, discrete_sampled), self.log_prob_continuous(continuous, continuous_sampled, selector = selector))

        return self.maybe_concat(output, concat = concat)

    def entropy_discrete(
        self,
        discrete_logits:  Tensor | list[Tensor] | tuple[Tensor, ...]
    ):
        is_list_tuple = isinstance(discrete_logits, (list, tuple))

        if not is_list_tuple:
            return calc_entropy(discrete_logits)

        assert len(discrete_logits) > 0

        if self.use_parallel_multi_discrete:
            discrete_logits, lens = cat_with_lens(discrete_logits)
            probs = segmented_softmax(discrete_logits, lens)

            neg_prob_log_prob = -probs * log(probs)
            neg_prob_log_prob = rearrange(neg_prob_log_prob, '... l -> l ...')

            entropies = torch.segment_reduce(neg_prob_log_prob, 'sum', lengths = lens)
        else:
            entropies = [calc_entropy(logit) for logit in discrete_logits]

        entropies = rearrange(entropies, 'nd ... -> ... nd')
        return entropies

    def entropy_continuous(
        self,
        continuous_dist_params,
        selector = None
    ):
        assert exists(selector)
        assert selector.continuous_log_var_embed
        dist = mean_log_var_to_normal_dist(continuous_dist_params)
        return dist.entropy()

    def entropy(
        self,
        dist,
        selector_index = None,
        concat = False
    ):
        selector = self.get_selector(selector_index)

        output = None

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                output = self.entropy_discrete(dist)

            elif selector.has_continuous:
                output = self.entropy_continuous(dist, selector = selector)

        else:
            discrete, continuous = dist
            output = DiscreteContinuous(self.entropy_discrete(discrete), self.entropy_continuous(continuous, selector = selector))

        return self.maybe_concat(output, concat = concat)

    def forward(
        self,
        embed,
        targets = None,
        return_loss = False,
        return_only_discrete_or_continuous = None,
        selector_index = None
    ):
        return_only_discrete_or_continuous = default(return_only_discrete_or_continuous, self.return_only_discrete_or_continuous)

        assert xnor(exists(targets), return_loss), '`target` must be passed in if `return_loss` set to True and vice versa'

        selector = self.get_selector(selector_index)

        # discrete unembedding

        discrete_logits_for_groups = None

        if selector.has_discrete:
            discrete_unembed = selector.discrete_selector.get_readout_embeds()
            all_discrete_logits = einsum(embed, discrete_unembed, '... d, nd d -> ... nd')

            discrete_logits_for_groups = selector.discrete_selector.split_packed(all_discrete_logits)

        # continuous unembedding

        continuous_dist_params = None

        if selector.has_continuous:
            continuous_unembed = selector.continuous_selector.get_mean_logvar_embed()
            continuous_dist_params = einsum(embed, continuous_unembed, '... d, nc d -> ... nc')

            if selector.continuous_log_var_embed:
                continuous_dist_params = rearrange(continuous_dist_params, '... (mu_logvar nc) -> ... nc mu_logvar', mu_logvar = 2)

        # maybe only return distribution parameters

        if not return_loss:
            if self.return_one_discrete_logits and selector.has_discrete and selector.discrete_selector.num_discrete_sets == 1 and exists(discrete_logits_for_groups):
                discrete_logits_for_groups = first(discrete_logits_for_groups)

            if selector.one_of_discrete_or_continuous and return_only_discrete_or_continuous:
                if selector.has_discrete:
                    return discrete_logits_for_groups

                if selector.has_continuous:
                    return continuous_dist_params

            return DiscreteContinuous(discrete_logits_for_groups, continuous_dist_params)

        # handle destructing of target

        discrete_targets = targets
        continuous_targets = targets

        if selector.has_discrete and selector.has_continuous:
            assert isinstance(targets, (tuple, list)) and len(targets) == 2
            discrete_targets, continuous_targets = targets

        # take care of only one discrete logit group, as in language modeling

        if self.return_one_discrete_logits and selector.has_discrete and selector.discrete_selector.num_discrete_sets == 1:
            discrete_targets = rearrange(discrete_targets, '... -> ... 1')

        # handle basic losses

        discrete_losses = self.zero

        if selector.has_discrete:
            if self.use_parallel_multi_discrete:
                log_probs = self.log_prob_discrete(discrete_logits_for_groups, discrete_targets)
                discrete_losses = -log_probs.sum(dim = -1).mean()
            else:
                discrete_losses = tuple(F.cross_entropy(rearrange(discrete_logit, 'b ... nd -> b nd ...'), one_target) for discrete_logit, one_target in zip(discrete_logits_for_groups, discrete_targets.unbind(dim = -1)))
                discrete_losses = sum(discrete_losses)

        continuous_losses = self.zero

        if selector.has_continuous:

            if selector.continuous_log_var_embed:
                gaussian = mean_log_var_to_normal_dist(continuous_dist_params)

                continuous_losses = -gaussian.log_prob(continuous_targets)
            else:
                continuous_losses = F.mse_loss(continuous_dist_params, continuous_targets, reduction = 'none')

            continuous_losses = reduce(continuous_losses, '... nc -> ...', 'sum')

            continuous_losses = continuous_losses.mean()

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                return discrete_losses

            if selector.has_continuous:
                return continuous_losses

        return DiscreteContinuous(discrete_losses, continuous_losses)

    def kl_div_discrete(
        self,
        discrete_logits_true: Tensor | list[Tensor] | tuple[Tensor, ...],
        discrete_logits_pred: Tensor | list[Tensor] | tuple[Tensor, ...]
    ):
        is_list_tuple = isinstance(discrete_logits_true, (list, tuple))

        if not is_list_tuple:
            discrete_logits_true = (discrete_logits_true,)
            discrete_logits_pred = (discrete_logits_pred,)

        assert len(discrete_logits_true) > 0
        assert len(discrete_logits_true) == len(discrete_logits_pred)

        if self.use_parallel_multi_discrete:
            discrete_logits_true, lens = cat_with_lens(discrete_logits_true)
            discrete_logits_pred, _ = cat_with_lens(discrete_logits_pred)

            probs_true = segmented_softmax(discrete_logits_true, lens)
            probs_pred = segmented_softmax(discrete_logits_pred, lens)

            kl = probs_true * (log(probs_true) - log(probs_pred))
            kl = rearrange(kl, '... l -> l ...')

            kl_divs = torch.segment_reduce(kl, 'sum', lengths = lens)
            kl_divs = rearrange(kl_divs, 'nd ... -> ... nd')
        else:
            kl_divs = []

            for logits_true, logits_pred in zip(discrete_logits_true, discrete_logits_pred):
                probs_true = logits_true.softmax(dim = -1)
                log_probs_true = logits_true.log_softmax(dim = -1)
                log_probs_pred = logits_pred.log_softmax(dim = -1)

                kl = F.kl_div(log_probs_pred, log_probs_true, reduction = 'none', log_target = True)
                kl_divs.append(kl.sum(dim = -1))

            kl_divs = stack(kl_divs, dim = -1)

        if not is_list_tuple:
            kl_divs = rearrange(kl_divs, '... 1 -> ...')

        return kl_divs

    def kl_div_continuous(
        self,
        continuous_dist_params_true,
        continuous_dist_params_pred,
        selector = None
    ):
        assert exists(selector)
        assert selector.continuous_log_var_embed

        dist_true = mean_log_var_to_normal_dist(continuous_dist_params_true)
        dist_pred = mean_log_var_to_normal_dist(continuous_dist_params_pred)

        return torch.distributions.kl.kl_divergence(dist_true, dist_pred)

    def kl_div(
        self,
        dist_true,
        dist_pred,
        selector_index = None
    ):
        selector = self.get_selector(selector_index)

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                return self.kl_div_discrete(dist_true, dist_pred)

            if selector.has_continuous:
                return self.kl_div_continuous(dist_true, dist_pred, selector = selector)

        discrete_true, continuous_true = dist_true
        discrete_pred, continuous_pred = dist_pred

        return DiscreteContinuous(
            self.kl_div_discrete(discrete_true, discrete_pred),
            self.kl_div_continuous(continuous_true, continuous_pred, selector = selector)
        )

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
