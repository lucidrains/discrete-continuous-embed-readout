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

def compact(arr):
    return list(filter(exists, arr))

def first(arr):
    return arr[0]

def is_unique(arr):
    return len(arr) == len(set(arr))

def xnor(x, y):
    return x == y

def default(v, d):
    return v if exists(v) else d

def flatten(arr):
    return [el for subarr in arr for el in subarr]

def atanh(t, eps = 1e-6):
    return 0.5 * (log(1 + t, eps = eps) - log(1 - t, eps = eps))

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

def softclamp(t, value = 15.):
    return (t / value).tanh() * value

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
    return mu + torch.randn_like(mu) * std * temperature

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
    assert len(dists) > 0, 'empty distributions'
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

class MultiCategorical:
    def __init__(
        self,
        logits: Tensor | list[Tensor] | tuple[Tensor, ...],
        use_parallel_multi_discrete = True,
        ignore_index = -1
    ):
        self.logits = logits
        self.ignore_index = ignore_index

        is_list_tuple = isinstance(logits, (list, tuple))

        first_tensor = first(logits) if is_list_tuple else logits
        is_mps = first_tensor.device.type == 'mps'

        self.use_parallel_multi_discrete = use_parallel_multi_discrete and not is_mps
        self._is_list_tuple = is_list_tuple

    @property
    def param(self):
        return self.logits

    def sample(
        self,
        temperature = 1.,
        eps = 1e-20
    ):
        # handle list or tuple of logits

        logits = self.logits

        if not self._is_list_tuple:
            logits = (logits,)

        if len(logits) > 1 and self.use_parallel_multi_discrete:
            sampled = gumbel_sample_multi_categorical(logits, temperature = temperature, eps = eps)
        else:
            sampled = tree_map_tensor(logits, partial(gumbel_sample, temperature = temperature, eps = eps))
            sampled = stack(sampled, dim = -1)

        if not self._is_list_tuple:
            sampled = rearrange(sampled, '... 1 -> ...')

        return sampled

    def log_prob(self, value):
        logits = self.logits

        if not self._is_list_tuple:
            # if single tensor passed in, but value has an extra dimension (sampled auto-squeezed), unsqueeze back
            logits = (logits,)

            if value.ndim == (logits[0].ndim - 1):
                value = rearrange(value, '... -> ... 1')

        assert len(logits) > 0, 'empty discrete logits'

        lens = tensor([d.shape[-1] for d in logits], device = first(logits).device)
        offsets = exclusive_cumsum(lens)

        indices = value + offsets

        # handle log softmax

        if self.use_parallel_multi_discrete:
            logits, lens = cat_with_lens(logits)
            log_softmaxed = log(segmented_softmax(logits, lens))
        else:
            log_softmaxed = [logit.log_softmax(dim = -1) for logit in logits]
            log_softmaxed = cat(log_softmaxed, dim = -1)

        # handle ignore index

        has_ignore_index = exists(self.ignore_index)

        if has_ignore_index:
             ignore_mask = value == self.ignore_index
             indices = indices.masked_fill(ignore_mask, 0)

        # gather log probs

        log_probs = log_softmaxed.gather(-1, indices)

        if has_ignore_index:
            log_probs = log_probs.masked_fill(ignore_mask, 0.)

        if not self._is_list_tuple:
            log_probs = rearrange(log_probs, '... 1 -> ...')

        return log_probs

    def entropy(self):
        logits = self.logits

        if not self._is_list_tuple:
            return calc_entropy(logits)

        assert len(logits) > 0, 'empty discrete logits'

        if self.use_parallel_multi_discrete:
            logits, lens = cat_with_lens(logits)
            probs = segmented_softmax(logits, lens)

            neg_prob_log_prob = -probs * log(probs)
            neg_prob_log_prob = rearrange(neg_prob_log_prob, '... l -> l ...')

            entropies = torch.segment_reduce(neg_prob_log_prob, 'sum', lengths = lens)
        else:
            entropies = [calc_entropy(logit) for logit in logits]

        entropies = rearrange(entropies, 'nd ... -> ... nd')
        return entropies

    def kl_div(self, other):
        logits_true = self.logits
        logits_pred = other.logits

        if not self._is_list_tuple:
            logits_true = (logits_true,)
            logits_pred = (logits_pred,)

        assert len(logits_true) > 0, 'empty logits'
        assert len(logits_true) == len(logits_pred), f'logits length mismatch: true {len(logits_true)} vs pred {len(logits_pred)}'

        if self.use_parallel_multi_discrete:
            logits_true, lens = cat_with_lens(logits_true)
            logits_pred, _ = cat_with_lens(logits_pred)

            probs_true = segmented_softmax(logits_true, lens)
            probs_pred = segmented_softmax(logits_pred, lens)

            kl = probs_true * (log(probs_true) - log(probs_pred))
            kl = rearrange(kl, '... l -> l ...')

            kl_divs = torch.segment_reduce(kl, 'sum', lengths = lens)
            kl_divs = rearrange(kl_divs, 'nd ... -> ... nd')
        else:
            kl_divs = []

            for l_true, l_pred in zip(logits_true, logits_pred):
                probs_true = l_true.softmax(dim = -1)
                log_probs_true = l_true.log_softmax(dim = -1)
                log_probs_pred = l_pred.log_softmax(dim = -1)

                kl = F.kl_div(log_probs_pred, log_probs_true, reduction = 'none', log_target = True)
                kl_divs.append(kl.sum(dim = -1))

            kl_divs = stack(kl_divs, dim = -1)

        if not self._is_list_tuple:
            kl_divs = rearrange(kl_divs, '... 1 -> ...')

        return kl_divs

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
        embeddings: nn.Embedding | None,
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
            assert indices.shape[-1] == self.num_discrete_sets, f'shape of input must end with {self.num_discrete_sets}, but got {indices.shape[-1]}'
            indices = indices + self.discrete_set_offsets

        embed_indices = self.discrete_indices[indices]

        if not exists(self.embeddings):
            raise RuntimeError('embeddings not present')

        return self.embeddings(embed_indices)

    def get_readout_embeds(
        self
    ):
        if not exists(self.embeddings):
            raise RuntimeError('embeddings not present')

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
        embed: nn.Embedding | None,
        num_continuous,
        embedding_offset,
        continuous_mean_std: Module | None,
        continuous_log_var_embed,
        continuous_squashed = False
    ):
        super().__init__()
        # embedding is [discrete] [continuous mean] [?continuous log var]

        continuous_indices = tensor(continuous_indices)
        assert continuous_indices.unique().numel() == continuous_indices.numel(), f'continuous indices must be unique, received {continuous_indices.tolist()}'

        self.embed = embed

        self.continuous_mean_std = None
        if exists(continuous_mean_std):
            self.continuous_mean_std = BufferModule(continuous_mean_std.data[continuous_indices])

        self.continuous_squashed = continuous_squashed
        self.continuous_log_var_embed = continuous_log_var_embed

        # offset by discrete

        continuous_indices = continuous_indices + embedding_offset

        if continuous_log_var_embed:
            continuous_log_var_indices = continuous_indices + num_continuous # offset by continuous mu

            continuous_indices = cat((continuous_indices, continuous_log_var_indices))

        self.register_buffer('continuous_indices', continuous_indices, persistent = False)

    def get_embed(self):
        if not exists(self.embed):
            raise RuntimeError('embeddings not present')

        return self.embed(self.continuous_indices)

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
        continuous_squashed = False
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
                continuous_log_var_embed = continuous_log_var_embed,
                continuous_squashed = continuous_squashed
            )

    @property
    def discrete_indices(self):
        return self.discrete_selector.discrete_indices if exists(self.discrete_selector) else None

    @property
    def continuous_mean_std(self):
        return self.continuous_selector.continuous_mean_std if exists(self.continuous_selector) else None

    @property
    def continuous_log_var_embed(self):
        return self.continuous_selector.continuous_log_var_embed if exists(self.continuous_selector) else None

    @property
    def continuous_squashed(self):
        return self.continuous_selector.continuous_squashed if exists(self.continuous_selector) else False

    # methods for inferring whether to return tuple or single value

    def validate_and_return_inputs(
        self,
        inp
    ):
        if not is_tensor(inp):
            return inp

        assert self.one_of_discrete_or_continuous, 'input validation only supported for single modality selectors'
        dtype = inp.dtype

        if dtype in (torch.int, torch.long) and self.has_discrete:
            return (inp, None)
        elif dtype == torch.float and self.has_continuous:
            return (None, inp)
        else:
            raise ValueError('invalid tensor')

# base

class Base(Module):

    @beartype
    def __init__(
        self,
        dim: int | None,
        num_discrete: int | tuple[int, ...] = 0,
        num_continuous: int = 0,
        selector: SelectorConfig | None = None,
        selectors: list[SelectorConfig] | None = None,
        continuous_log_var_embed = True,
        continuous_mean_std: Tensor | None = None,
        use_parallel_multi_discrete = True,
        return_only_discrete_or_continuous = True,
        continuous_squashed = False,
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

            assert len(selectors) > 0, 'empty selectors'

            # normalize selectors

            selectors_configs = []

            for selector in selectors:

                discrete_indices, continuous_indices = self._process_selector_config(selector)

                if exists(discrete_indices):
                    max_discrete_index = max(max_discrete_index, max(flatten(discrete_indices)))

                if exists(continuous_indices):
                    max_continuous_index = max(max_continuous_index, *continuous_indices)

                selectors_configs.append((discrete_indices, continuous_indices))

            if max_discrete_index >= 0:
                assert max_discrete_index < total_discrete, f'discrete index out of bounds: max index {max_discrete_index} >= total discrete embeddings {total_discrete}'

            if max_continuous_index >= 0:
                assert max_continuous_index < num_continuous, f'continuous index out of bounds: max index {max_continuous_index} >= num continuous {num_continuous}'

        # infer has discrete or continuous

        self.has_discrete = total_discrete > 0
        self.has_continuous = num_continuous > 0

        self.has_continuous = num_continuous > 0

        assert total > 0, 'cannot have both discrete and continuous disabled'

        self.dim = dim

        # all embeddings for discrete and continuous stored together
        # order will be [discrete] [continuous]
        # discrete is further broken up by groups if tuple of ints passed in - so [discrete group 1] [discrete group 2] ... [continuous]

        if exists(dim) and dim > 0:
            self.embeddings = nn.Embedding(total, dim)
            nn.init.normal_(self.embeddings.weight, std = 1e-2)
        else:
            self.embeddings = None

        # maybe norm and inverse norm

        self.can_norm_continuous = exists(continuous_mean_std)

        if self.can_norm_continuous:
            assert self.has_continuous, 'continuous mean std given but no continuous dims'
            assert continuous_mean_std.shape == (num_continuous, 2), f'continuous mean std shape mismatch, expected ({num_continuous}, 2) but got {continuous_mean_std.shape}'
            assert (continuous_mean_std[..., -1] > 0).all(), 'std must be positive'

            continuous_mean_std = BufferModule(continuous_mean_std)

        # continuous action range

        self.continuous_squashed = continuous_squashed

        # discrete related computed values

        self.use_parallel_multi_discrete = use_parallel_multi_discrete # sampling, entropy, log prob in parallel for multi-discrete

        # handle selectors

        self.selectors = ModuleList([])

        if not has_selectors:
            counter = count(0)
            default_discrete_indices = [[next(counter) for _ in range(n)] for n in num_discrete] if self.has_discrete else None
            default_continuous_indices = arange(num_continuous).tolist() if self.has_continuous else None

            selectors_configs = [(default_discrete_indices, default_continuous_indices)]

        self.continuous_log_var_embed = continuous_log_var_embed
        self.num_continuous = num_continuous
        self.embedding_offset = total_discrete
        self.continuous_mean_std = continuous_mean_std

        for selector_config in selectors_configs:
            discrete_indices, continuous_indices = selector_config

            selector = self.create_discrete_continuous_selector(
                discrete_indices = discrete_indices,
                continuous_indices = continuous_indices,
                continuous_squashed = continuous_squashed
            )

            self.selectors.append(selector)

        self.num_discrete_sets = len(num_discrete)

        # delegation properties

        self.return_only_discrete_or_continuous = return_only_discrete_or_continuous

        # epsilon

        self.eps = eps

    def create_discrete_continuous_selector(
        self,
        discrete_indices = None,
        continuous_indices = None,
        continuous_squashed = False
    ):
        return DiscreteContinuousSelector(
            embeddings = self.embeddings,
            discrete_set_indices = discrete_indices,
            continuous_indices = continuous_indices,
            num_continuous = self.num_continuous,
            embedding_offset = self.embedding_offset,
            continuous_log_var_embed = self.continuous_log_var_embed,
            continuous_mean_std = self.continuous_mean_std,
            continuous_squashed = continuous_squashed
        )

    def _process_selector_config(self, selector: SelectorConfig):
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
                assert len(group) > 0, 'empty discrete group'



        assert not exists(discrete_indices) or all([is_unique(indices) for indices in discrete_indices])
        assert not exists(continuous_indices) or is_unique(continuous_indices)

        return discrete_indices, continuous_indices

    def get_selector(
        self,
        selector_index: int | None = None,
        selector_config: SelectorConfig | None = None
    ):
        if exists(selector_config):
            discrete_indices, continuous_indices = self._process_selector_config(selector_config)
            return self.create_discrete_continuous_selector(discrete_indices = discrete_indices, continuous_indices = continuous_indices)

        if len(self.selectors) == 1:
            return self.selectors[0]

        assert exists(selector_index), 'selector index required'
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
        selector_index: int | None = None,
        selector_config: SelectorConfig | None = None,
        concat_discrete_continuous = False
    ):
        return_only_discrete_or_continuous = default(return_only_discrete_or_continuous, self.return_only_discrete_or_continuous)

        assert not (normalize_continuous and not self.can_norm_continuous), 'cannot normalize continuous without mean/std'

        selector = self.get_selector(selector_index, selector_config = selector_config)

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
        auto_squeeze_single_output = True,
        ignore_index = -1,
        continuous_softclamp_logvar: float | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.auto_squeeze_single_output = auto_squeeze_single_output
        self.return_one_discrete_logits = default(return_one_discrete_logits, self.num_discrete_sets == 1)
        self.continuous_softclamp_logvar = continuous_softclamp_logvar
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

        dist = MultiCategorical(
            logits = discrete_logits,
            use_parallel_multi_discrete = self.use_parallel_multi_discrete
        )

        sampled = dist.sample(temperature = temperature, eps = self.eps)

        if not is_list_tuple and self.auto_squeeze_single_output:
            sampled = rearrange(sampled, '... 1 -> ...')

        return sampled

    def sample_continuous(
        self,
        continuous_dist_params,
        temperature = 1.,
        selector = None
    ):
        assert exists(selector), 'selector required for continuous sampling'
        assert selector.continuous_log_var_embed, 'continuous log var embed required'

        sampled = gaussian_sample(continuous_dist_params, temperature)

        if selector.continuous_squashed:
            sampled = sampled.tanh()

        if not self.can_norm_continuous:
            return sampled

        mean, std = selector.continuous_mean_std.data.unbind(dim = -1)
        inverse_normed = sampled * std + mean
        return inverse_normed

    def sample(
        self,
        dist,
        selector_index: int | None = None,
        selector_config: SelectorConfig | None = None
    ):
        selector = self.get_selector(selector_index, selector_config = selector_config)

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
        dist = MultiCategorical(
            logits = discrete_logits,
            use_parallel_multi_discrete = self.use_parallel_multi_discrete,
            ignore_index = self.ignore_index
        )

        return dist.log_prob(sampled)

    def log_prob_continuous(
        self,
        continuous_dist_params,
        sampled,
        selector = None
    ):
        assert exists(selector)
        assert selector.continuous_log_var_embed

        gaussian_sampled = atanh(sampled, eps = self.eps) if selector.continuous_squashed else sampled

        dist = mean_log_var_to_normal_dist(continuous_dist_params)
        log_prob = dist.log_prob(gaussian_sampled)

        if not selector.continuous_squashed:
            return log_prob

        return log_prob - 2 * (log(tensor(2.)) - gaussian_sampled - F.softplus(-2 * gaussian_sampled))

    def maybe_concat(self, output, concat = False):
        if not concat:
            return output

        if isinstance(output, DiscreteContinuous):
            output = (output.discrete, output.continuous)

        output = cast_tuple(output)
        output = compact(output)

        if len(output) == 0:
            return None

        # if any tensor is (batch, seq) - assume it is single discrete and unsqueeze (so it becomes (batch, seq, 1))

        output = [rearrange(t, '... -> ... 1') if (t.ndim == 2 and self.return_one_discrete_logits) else t for t in output]
        return cat(output, dim = -1)

    def log_prob(
        self,
        logits,
        sampled,
        selector_index: int | None = None,
        selector_config: SelectorConfig | None = None,
        concat = False
    ):
        selector = self.get_selector(selector_index, selector_config = selector_config)

        output = None

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                output = self.log_prob_discrete(logits, sampled)

            elif selector.has_continuous:
                output = self.log_prob_continuous(logits, sampled, selector = selector)

        else:
            discrete, continuous = logits
            discrete_sampled, continuous_sampled = sampled
            output = DiscreteContinuous(self.log_prob_discrete(discrete, discrete_sampled), self.log_prob_continuous(continuous, continuous_sampled, selector = selector))

        return self.maybe_concat(output, concat = concat)

    def entropy_discrete(
        self,
        discrete_logits:  Tensor | list[Tensor] | tuple[Tensor, ...]
    ):
        dist = MultiCategorical(
            logits = discrete_logits,
            use_parallel_multi_discrete = self.use_parallel_multi_discrete
        )

        return dist.entropy()

    def entropy_continuous(
        self,
        continuous_dist_params,
        selector = None
    ):
        assert exists(selector), 'selector required'
        assert selector.continuous_log_var_embed, 'continuous log var embed required'

        if selector.continuous_squashed:
            return None

        dist = mean_log_var_to_normal_dist(continuous_dist_params)
        return dist.entropy()

    def entropy(
        self,
        logits,
        selector_index: int | None = None,
        selector_config: SelectorConfig | None = None,
        concat = False
    ):
        selector = self.get_selector(selector_index, selector_config = selector_config)

        output = None

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                output = self.entropy_discrete(logits)

            elif selector.has_continuous:
                output = self.entropy_continuous(logits, selector = selector)

        else:
            discrete, continuous = logits
            output = DiscreteContinuous(self.entropy_discrete(discrete), self.entropy_continuous(continuous, selector = selector))

        return self.maybe_concat(output, concat = concat)

    def calculate_loss(
        self,
        logits,
        targets,
        selector_index: int | None = None,
        selector_config: SelectorConfig | None = None,
        mask = None,
        return_unreduced_loss = False
    ):
        selector = self.get_selector(selector_index, selector_config=selector_config)

        # handle destructuring of logits

        discrete_logits = logits
        continuous_dist_params = logits

        if selector.has_discrete and selector.has_continuous:
            assert isinstance(logits, (tuple, list, DiscreteContinuous)) and len(logits) == 2, f'logits must be tuple of (discrete, continuous) when both are present, received {type(logits)}'
            discrete_logits, continuous_dist_params = logits

        # handle destructuring of targets

        discrete_targets = targets
        continuous_targets = targets

        if selector.has_discrete and selector.has_continuous:
            assert isinstance(targets, (tuple, list, DiscreteContinuous)) and len(targets) == 2, f'targets must be tuple of (discrete, continuous) when both are present, received {type(targets)}'
            discrete_targets, continuous_targets = targets

        # take care of only one discrete logit group, as in language modeling

        if self.return_one_discrete_logits and selector.has_discrete and selector.discrete_selector.num_discrete_sets == 1 and self.auto_squeeze_single_output:
            discrete_targets = rearrange(discrete_targets, '... -> ... 1')

        # calculations

        discrete_losses = self.zero

        if selector.has_discrete:
            if self.use_parallel_multi_discrete:
                log_probs = self.log_prob_discrete(discrete_logits, discrete_targets)
                discrete_losses = -log_probs
            else:
                discrete_losses = []

                discrete_logits = cast_tuple(discrete_logits)

                for discrete_logit, one_target in zip(discrete_logits, discrete_targets.unbind(dim = -1)):

                    discrete_loss = F.cross_entropy(
                        rearrange(discrete_logit, 'b ... c -> b c ...'),
                        one_target,
                        reduction = 'none',
                        ignore_index = self.ignore_index
                    )

                    discrete_losses.append(discrete_loss)

                if len(discrete_losses) > 1:
                    discrete_losses = stack(discrete_losses, dim = -1)
                else:
                    discrete_losses = first(discrete_losses)

        continuous_losses = self.zero

        if selector.has_continuous:
            if selector.continuous_log_var_embed:
                gaussian = mean_log_var_to_normal_dist(continuous_dist_params)
                continuous_losses = -gaussian.log_prob(continuous_targets)
            else:
                continuous_losses = F.mse_loss(continuous_dist_params, continuous_targets, reduction = 'none')

        # handle masking

        if exists(mask):
            if selector.has_discrete:
                discrete_mask = mask
                if discrete_losses.ndim == (mask.ndim + 1):
                    discrete_mask = rearrange(mask, '... -> ... 1')

                discrete_losses = discrete_losses * discrete_mask

            if selector.has_continuous:
                continuous_mask = mask
                if continuous_losses.ndim == (mask.ndim + 1):
                    continuous_mask = rearrange(mask, '... -> ... 1')

                continuous_losses = continuous_losses * continuous_mask

        # return early if unreduced

        if return_unreduced_loss:
            if selector.one_of_discrete_or_continuous:
                if selector.has_discrete:
                    return discrete_losses

                if selector.has_continuous:
                    return continuous_losses

            return DiscreteContinuous(discrete_losses, continuous_losses)

        # reduce

        if selector.has_discrete:
            discrete_divisor = mask.sum() if exists(mask) else tensor(discrete_losses.numel(), device = self.zero.device)

            if exists(self.ignore_index):
                valid_ignore_mask = (discrete_targets != self.ignore_index)

                if valid_ignore_mask.ndim > discrete_losses.ndim:
                    valid_ignore_mask = valid_ignore_mask.any(dim = -1)

                # if the losses have an extra action dimension, but the targets do not (which is the case for single discrete action with auto_squeeze), we need to expand the ignore mask

                if valid_ignore_mask.ndim < discrete_losses.ndim:
                    valid_ignore_mask = repeat(valid_ignore_mask, '... -> ... 1')

                if exists(mask):
                    if valid_ignore_mask.ndim == (mask.ndim + 1):
                         mask = rearrange(mask, '... -> ... 1')

                    valid_ignore_mask = valid_ignore_mask & mask

                discrete_divisor = valid_ignore_mask.sum()

            discrete_losses = discrete_losses.sum() / discrete_divisor.clamp_min(1.)

        if selector.has_continuous:
            continuous_divisor = mask.sum() if exists(mask) else tensor(continuous_losses.numel(), device = self.zero.device)
            continuous_losses = continuous_losses.sum() / continuous_divisor.clamp_min(1.)

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                return discrete_losses

            if selector.has_continuous:
                return continuous_losses

        return DiscreteContinuous(discrete_losses, continuous_losses)

    def forward(
        self,
        embed,
        targets = None,
        return_loss = False,
        return_unreduced_loss = False,
        loss_mask = None,
        return_only_discrete_or_continuous = None,
        selector_index = None,
        selector_config = None
    ):
        return_only_discrete_or_continuous = default(return_only_discrete_or_continuous, self.return_only_discrete_or_continuous)

        assert xnor(exists(targets), return_loss), '`target` must be passed in if `return_loss` set to True and vice versa'

        selector = self.get_selector(selector_index, selector_config = selector_config)

        # discrete unembedding

        discrete_logits_for_groups = None

        if selector.has_discrete:
            discrete_unembed = selector.discrete_selector.get_readout_embeds()
            all_discrete_logits = einsum(embed, discrete_unembed, '... d, nd d -> ... nd')

            discrete_logits_for_groups = selector.discrete_selector.split_packed(all_discrete_logits)

        # continuous unembedding

        continuous_dist_params = None

        if selector.has_continuous:
            continuous_unembed = selector.continuous_selector.get_embed()
            continuous_dist_params = einsum(embed, continuous_unembed, '... d, nc d -> ... nc')

            if selector.continuous_log_var_embed:
                continuous_dist_params = rearrange(continuous_dist_params, '... (mu_logvar nc) -> ... nc mu_logvar', mu_logvar = 2)

                if exists(self.continuous_softclamp_logvar):
                    mu, log_var = continuous_dist_params.unbind(dim = -1)
                    log_var = softclamp(log_var, self.continuous_softclamp_logvar)
                    continuous_dist_params = stack((mu, log_var), dim = -1)

        # maybe only return distribution parameters

        if not return_loss:
            if self.return_one_discrete_logits and selector.has_discrete and selector.discrete_selector.num_discrete_sets == 1 and exists(discrete_logits_for_groups) and self.auto_squeeze_single_output:
                discrete_logits_for_groups = first(discrete_logits_for_groups)

            if selector.one_of_discrete_or_continuous and return_only_discrete_or_continuous:
                if selector.has_discrete:
                    return discrete_logits_for_groups

                if selector.has_continuous:
                    return continuous_dist_params

            return DiscreteContinuous(discrete_logits_for_groups, continuous_dist_params)

        # handle basic losses

        logits = DiscreteContinuous(discrete_logits_for_groups, continuous_dist_params)

        if selector.one_of_discrete_or_continuous:
            if selector.has_discrete:
                logits = discrete_logits_for_groups
            elif selector.has_continuous:
                logits = continuous_dist_params

        return self.calculate_loss(
            logits,
            targets,
            selector_index = selector_index,
            selector_config = selector_config,
            mask = loss_mask,
            return_unreduced_loss = return_unreduced_loss
        )

    def kl_div_discrete(
        self,
        discrete_logits_true: Tensor | list[Tensor] | tuple[Tensor, ...],
        discrete_logits_pred: Tensor | list[Tensor] | tuple[Tensor, ...]
    ):
        dist_true = MultiCategorical(
            logits = discrete_logits_true,
            use_parallel_multi_discrete = self.use_parallel_multi_discrete
        )

        dist_pred = MultiCategorical(
            logits = discrete_logits_pred,
            use_parallel_multi_discrete = self.use_parallel_multi_discrete
        )

        kl_divs = dist_true.kl_div(dist_pred)

        # handle single output auto squeeze logic if needed, although MultiCategorical handles single logic internally
        # we need to respect the Readout auto_squeeze_single_output logic if passing single tensor

        is_list_tuple = isinstance(discrete_logits_true, (list, tuple))
        if not is_list_tuple and self.auto_squeeze_single_output and kl_divs.shape[-1] == 1:
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

        assert not selector.continuous_squashed, 'kl divergence not supported for squashed gaussian'

        dist_true = mean_log_var_to_normal_dist(continuous_dist_params_true)
        dist_pred = mean_log_var_to_normal_dist(continuous_dist_params_pred)

        return torch.distributions.kl.kl_divergence(dist_true, dist_pred)

    def kl_div(
        self,
        dist_true,
        dist_pred,
        selector_index = None,
        selector_config = None
    ):
        selector = self.get_selector(selector_index, selector_config = selector_config)

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

class ParameterlessReadout(Readout):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        assert 'dim' not in kwargs
        super().__init__(*args, dim = 0, **kwargs)

# helper functions for creating both, with optional weight tying

def EmbedAndReadout(
    *args,
    weight_tie = False,
    embed_kwargs: dict = dict(),
    readout_kwargs: dict = dict(),
    explicit_single_action_dim_given: bool | None = None,
    **kwargs,
):
    if exists(explicit_single_action_dim_given):
        assert 'auto_append_discrete_set_dim' not in embed_kwargs, 'cannot pass `auto_append_discrete_set_dim` if `explicit_single_action_dim_given` is set'
        assert 'auto_squeeze_single_output' not in readout_kwargs, 'cannot pass `auto_squeeze_single_output` if `explicit_single_action_dim_given` is set'

        auto_unsqueeze_and_squeeze = not explicit_single_action_dim_given

        embed_kwargs = {**embed_kwargs, 'auto_append_discrete_set_dim': auto_unsqueeze_and_squeeze}
        readout_kwargs = {**readout_kwargs, 'auto_squeeze_single_output': auto_unsqueeze_and_squeeze}

    embed = Embed(*args, **embed_kwargs, **kwargs)
    readout = Readout(*args, **readout_kwargs, **kwargs)

    if weight_tie:
        embed.embeddings = readout.embeddings # readout has the superset

    return embed, readout
