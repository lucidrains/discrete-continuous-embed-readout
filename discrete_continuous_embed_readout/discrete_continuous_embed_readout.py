from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from torch.distributions import Normal

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

# classes

class Embed(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.dim = dim

class Readout(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.dim = dim
