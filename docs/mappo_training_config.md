# MAPPO Training Configuration Analysis

## Checkpoint Information


## Training Duration & Scale

| Metric | Value |
|--------|-------|
| **Training Iterations** | 8,030 iterations |
| **Total Environment Steps** | **8,363,244 steps** (~8.36M) |
| **Checkpoint Used for Eval** | checkpoint_000079 (iteration 79) |
| **Training Time** | ~22 hours |

---

## Learning Rate Configuration

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | `5e-05` (0.00005) |
| **LR Schedule** | **None** (constant) |
| **LR Decay** | **No** |

### Analysis:
- Learning rate of `5e-05` is **very conservative**
- Typical PPO uses `3e-4` or `5e-4` (6-10x higher)
- No learning rate annealing or decay throughout training
- Constant LR may prevent fine-tuning in later stages

---

## Core Training Hyperparameters

### PPO-Specific
```python
clip_param = 0.3              # PPO clipping parameter (default: 0.2)
kl_coeff = 0.2                # KL divergence coefficient
kl_target = 0.01              # Target KL for adaptive coefficient
use_kl_loss = True            # Use KL loss in addition to clipping
```

### Batch Sizes & Epochs
```python
train_batch_size = 4000       # Total samples per training iteration
minibatch_size = 64           # Minibatch size for gradient updates
num_epochs = 10               # Number of passes over each batch
sgd_minibatch_size = -1       # (deprecated parameter)
```

**Calculation**: 
- 4000 steps / 64 minibatch = **62.5 minibatches per epoch**
- 62.5 × 10 epochs = **625 gradient updates per iteration**

### Value Function
```python
gamma = 0.99                  # Discount factor
lambda = 1.0                  # GAE lambda (1.0 = full Monte Carlo returns)
use_gae = True                # Use Generalized Advantage Estimation
use_critic = True             # Use value function critic
vf_loss_coeff = 1.0           # Value function loss coefficient
vf_clip_param = 10.0          # Value function clipping parameter
```

### Regularization
```python
entropy_coeff = 0.0           # ⚠️ NO entropy regularization
entropy_coeff_schedule = None # No entropy schedule
grad_clip = None              # ⚠️ NO gradient clipping
grad_clip_by = "global_norm"  # Method (unused since grad_clip=None)
```

---

## Model Architecture

### Convolutional Encoder (for 7×7 local observations)
```python
conv_filters = [
    [32, [3, 3], 1],   # 32 filters, 3×3 kernel, stride 1
    [64, [3, 3], 1],   # 64 filters, 3×3 kernel, stride 1
    [128, [3, 3], 1],  # 128 filters, 3×3 kernel, stride 1
]
conv_activation = "relu"
```

### Fully Connected Layers
```python
fcnet_hiddens = [128, 128]    # Two hidden layers with 128 units each
fcnet_activation = "relu"
```

### Centralized Critic (MAPPO)
```python
# Custom value function that uses global state
central_value_head = Sequential(
    Linear(N * H * W * C, 128),  # N=2 agents, H=7, W=7, C=3 channels
    ReLU(),
    Linear(128, 1)
)
```

**Total Parameters**: 130,886 trainable parameters

---

## Compute Resources

### Parallel Sampling
```python
num_env_runners = 5           # Number of parallel rollout workers
num_envs_per_env_runner = 8   # Vectorized environments per worker
num_cpus_per_env_runner = 2   # CPU cores per worker
```

**Total parallel environments**: 5 workers × 8 envs = **40 environments**

### Hardware
```python
num_gpus = 1                  # Single GPU for learner
num_gpus_per_env_runner = 0   # Workers run on CPU
framework = "torch"           # PyTorch backend
```

---

## Environment Configuration

### Training Environment
```python
env = pursuit_v4.parallel_env(
    n_pursuers = 2,           # 2 pursuer agents
    n_evaders = 1,            # 1 evader
    freeze_evaders = True,    # ⚠️ Evaders are frozen during training
    x_size = 8,               # 8×8 grid
    y_size = 8,
    n_catch = 2,              # Both pursuers must overlap evader
    surround = False,         # Overlap mode (not surround)
    shared_reward = False,    # Individual rewards
    max_cycles = 100,         # Max episode length
)
```

### Preprocessing
```python
# Applied wrappers:
ss.pad_observations_v0(env)   # Pad obs to uniform shape
ss.pad_action_space_v0(env)   # Pad action space to uniform shape

# NOT applied (commented out in training script):
# ss.color_reduction_v0(env, mode="B")  # Grayscale
# ss.dtype_v0(env, "float32")           # Float conversion
# ss.normalize_obs_v0(env)              # Normalization
# ss.frame_stack_v1(env, 4)             # Frame stacking
```

---

## Policy Configuration

### Multi-Agent Setup
```python
policies = {
    "shared_pursuer": (None, None, None, {})
}

def policy_mapping_fn(agent_id, episode, **kwargs):
    return "shared_pursuer"  # All agents share one policy
```

**Policy Type**: Parameter-sharing (centralized training, decentralized execution)

---

## Key Training Characteristics

### ✅ Strengths
1. **Large batch size** (4000 steps) for stable gradients
2. **Multiple epochs** (10) per batch for sample efficiency
3. **Centralized critic** using global state (MAPPO architecture)
4. **Parallel sampling** (40 environments) for fast data collection
5. **Adequate training time** (8.36M steps)

### ⚠️ Potential Issues
1. **Very low learning rate** (`5e-05` vs typical `3e-4`)
   - May lead to slow convergence
   - Could underfit the problem space
   
2. **No entropy regularization** (`entropy_coeff = 0.0`)
   - Discourages exploration
   - Can lead to premature convergence
   - May produce overly deterministic policies
   
3. **No gradient clipping** (`grad_clip = None`)
   - Risk of training instability
   - Large gradients could destabilize learning
   
4. **No learning rate decay**
   - Misses opportunity for fine-tuning
   - Could benefit from annealing schedule
   
5. **High GAE lambda** (`lambda = 1.0`)
   - Full Monte Carlo returns
   - Higher variance in advantage estimates
   
6. **Trained only on frozen evaders**
   - Never saw moving targets during training
   - Surprisingly generalizes well to moving evaders (100% success!)


---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Total Training Steps** | 8,363,244 |
| **Training Iterations** | 8,030 |
| **Gradient Updates** | ~5,018,750 (625 per iter × 8030) |
| **Episodes Collected** | ~83,632 (assuming avg 100 steps/ep) |
| **Training Time** | ~22 hours |
| **Final Checkpoint** | checkpoint_000079 |



