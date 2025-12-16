## Discrete Continuous Embed Readout

Embedding and readout for simple categorical and gaussian distributions, from the language model to sophisticated robotic action spaces

## Install

```bash
pip install discrete-continuous-embed-readout
```

## Usage

### Discrete

For standard autoregressive language modeling or discrete action spaces.

```python
import torch
from discrete_continuous_embed_readout import EmbedAndReadout

# 1. Initialize

embed_readout = EmbedAndReadout(
    dim = 512,
    num_discrete = 20000              # vocabulary size
)

embed, readout = embed_readout

# 2. Embed

ids = torch.randint(0, 20000, (2, 1024))

embeds = embed(ids) # (2, 1024, 512)

# ... pass through your transformer / network ...

# 3. Readout

logits = readout(embeds) # (2, 1024, 20000)

# Calculate loss (automatically handles cross entropy)

labels = torch.randint(0, 20000, (2, 1024))

loss = readout(embeds, labels, return_loss = True)
loss.backward()

# Sampling and other utilities

sampled = readout.sample(logits)                # (2, 1024)
log_probs = readout.log_prob(logits, sampled)   # (2, 1024)
entropy = readout.entropy(logits)               # (2, 1024)
```

### Continuous

For continuous control or regression tasks.

```python
import torch
from discrete_continuous_embed_readout import EmbedAndReadout

# 1. Initialize

embed_readout = EmbedAndReadout(
    dim = 512,
    num_continuous = 4,              # 4 continuous dimensions
    continuous_mean_std = torch.ones(4, 2) # optional mean and std for normalization
)

embed, readout = embed_readout

# 2. Embed

values = torch.randn(2, 1024, 4)

embeds = embed(values) # (2, 1024, 512)

# ... pass through network ...

# 3. Readout (returns distinct Gaussian parameters)

dist_params = readout(embeds) # (2, 1024, 4, 2) - mean and log var

# Loss (Gaussian NLL)

targets = torch.randn(2, 1024, 4)

loss = readout(embeds, targets, return_loss = True)
loss.backward()

# Sampling

sampled = readout.sample(dist_params)               # (2, 1024, 4)
```

### Mixed Discrete and Continuous

For complex environments with both discrete and continuous action spaces.

```python
import torch
from discrete_continuous_embed_readout import EmbedAndReadout

# 1. Initialize

embed_readout = EmbedAndReadout(
    dim = 512,
    num_discrete = 100,
    num_continuous = 4
)

embed, readout = embed_readout

# 2. Embed inputs (passed as tuple)

discrete_in = torch.randint(0, 100, (2, 32))
continuous_in = torch.randn(2, 32, 4)

embeds = embed((discrete_in, continuous_in)) # (2, 32, 512)

# ... network ...

# 3. Readout

output = readout(embeds)

# Access individual logits/params
print(output.discrete.shape)   # (2, 32, 100)
print(output.continuous.shape) # (2, 32, 4, 2)

# Sampling returns tuple

sampled_discrete, sampled_continuous = readout.sample(output)
```

### Multi-Discrete

For action spaces with multiple independent discrete actions.

```python
import torch
from discrete_continuous_embed_readout import EmbedAndReadout

embed_readout = EmbedAndReadout(
    dim = 512,
    num_discrete = (10, 5, 8),    # 3 independent discrete actions
    use_parallel_multi_discrete = True # optimized parallel processing
)

embed, readout = embed_readout

# Input shape: (batch, seq, 3)
action_indices = torch.randint(0, 5, (2, 16, 3))

embeds = embed(action_indices)

# Readout returns list of logits if not using parallel optimization, or a special structure if so.
# However, the wrapper handles it seamlessly.

logits = readout(embeds)
sampled = readout.sample(logits) # (2, 16, 3)
```

### Runtime Selectors

You can also define inputs dynamically at runtime if your architecture shares embeddings across different modalities.

```python
discrete_config = [[0, 1, 2]] # specific indices for this input
continuous_config = [0, 1]    # specific continuous indices

embeds = embed(
    (discrete_in, continuous_in),
    selector_config = (discrete_config, continuous_config)
)
```
