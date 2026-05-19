# models -- Neural Network Components

This package contains all neural network modules for the ESC pipeline: the policy network, value network, transition model, and the Qwen backbone stub. All modules are implemented in PyTorch and follow a common design principle: they consume flat state vectors or structured `ESCState` / `ESCAction` inputs and produce tensors that feed into the MDP or the loss computation.

---

## Files

### `policy.py` -- PolicyNetwork

`PolicyNetwork` maps a flat state vector `s in R^{state_dim}` to a probability distribution over the action space:

```
pi_theta(a | s): R^{state_dim} -> Delta^{action_dim}
```

**Architecture:**

```
SharedTrunk(state_dim, hidden_dim)
  Linear(state_dim, hidden_dim) -> LayerNorm(hidden_dim) -> ReLU
  Linear(hidden_dim, hidden_dim) -> LayerNorm(hidden_dim) -> ReLU

Policy head:
  Linear(hidden_dim, action_dim) -> Softmax(dim=-1)
```

LayerNorm after each linear layer stabilizes training with high-dimensional inputs (the state vector is 1283-dimensional) and reduces sensitivity to weight initialization.

The `forward(state)` method accepts both unbatched `[state_dim]` and batched `[B, state_dim]` inputs; the softmax is applied along the last dimension in both cases.

`num_actions` is stored as an attribute so callers can verify the network's output size before creating a mismatch with a filtered candidate list.

---

### `value.py` -- ValueNetwork

`ValueNetwork` maps a flat state vector to a scalar value estimate in [-1, 1]:

```
V_theta(s): R^{state_dim} -> [-1, 1]
```

**Architecture:**

```
SharedTrunk(state_dim, hidden_dim)
  Linear(state_dim, hidden_dim) -> LayerNorm(hidden_dim) -> ReLU
  Linear(hidden_dim, hidden_dim) -> LayerNorm(hidden_dim) -> ReLU

Value head:
  Linear(hidden_dim, 1) -> Tanh
```

The tanh output activation bounds the estimate to (-1, 1). This matches the normalized reward structure: `R_cause` and `R_emotion` are designed to lie approximately in [-1, 1], and `R_phase` is in {0, 1}, so the cumulative episode return over a 20-turn horizon falls in a bounded range that tanh can represent without saturation.

`value(state_tensor)` is a convenience wrapper that disables gradient computation and returns a Python float. It is called by `MCTS.ValueNetwork.value()` during the evaluation phase of each simulation.

Both `PolicyNetwork` and `ValueNetwork` share the same `_SharedTrunk` architecture. In `mcts/mcts.py` these two heads share a single trunk instance (parameter efficiency during search), while in `models/` they are standalone modules optimized independently during training.

---

### `transition.py` -- TransitionModel

The transition model predicts three components of the next state from the current state and action:

```
f_theta(s_t, a_t) -> TransitionOutput(next_emotion, next_phase_logits, delta_resolution)
```

**`TransitionOutput`** is a dataclass with three fields:

| Field | Shape | Description |
|---|---|---|
| `next_emotion` | [D_E] | Predicted emotion vector for s_{t+1}, values in (-1, 1) via tanh |
| `next_phase_logits` | [3] | Raw logits over the three ESC phases (softmax applied in ESCEnv.step()) |
| `delta_resolution` | [N_C] | Per-cause resolution change in (-1, 1) via tanh |

**`TransitionModel`** is an abstract base class with a single abstract method `forward(state, action) -> TransitionOutput`. This abstraction allows `ESCEnv` to accept any transition model without knowing its internals.

**`LinearTransitionModel(TransitionModel, nn.Module)`** is the learned implementation.

Input construction:

```
action_vec = concat(one_hot(strategy_id, N_strategies), one_hot(cause_index, N_C))
input      = concat(state.to_tensor(), action_vec)
           in R^{state_dim + N_strategies + N_C} = R^{1283 + 8 + 4} = R^{1295}
```

The trunk processes this input through two LayerNorm-ReLU layers, then three separate heads predict each output component:

```
emotion_head:     tanh(Linear(hidden_dim, D_E))       -> next_emotion in (-1,1)^{D_E}
phase_head:       Linear(hidden_dim, 3)               -> next_phase_logits in R^3
resolution_head:  tanh(Linear(hidden_dim, N_C))       -> delta_resolution in (-1,1)^{N_C}
```

The tanh bounds on `next_emotion` and `delta_resolution` are important: without them, the resolution deltas could push cause probabilities far outside [0, 1] before the clamping in `ESCEnv.step()` corrects them, creating a degenerate training signal.

**`RandomTransitionModel`** is a non-learned baseline that adds small Gaussian noise to the emotion vector and a small positive random delta to resolution probabilities. It is used for:

- Pipeline validation before training is complete
- Ablation experiments that test whether learned dynamics are necessary
- Unit tests that need deterministic non-trivial dynamics (via a seeded `torch.Generator`)

Parameters:
- `emotion_noise_scale` (default 0.05): std of Gaussian noise added to emotion
- `resolution_delta_scale` (default 0.1): maximum absolute resolution delta per step
- `seed`: optional integer for reproducible stochastic dynamics

---

### `backbone_qwen.py` -- QwenBackbone

`QwenBackbone` is the integration stub for the Qwen-1.5-9B-Chat backbone model. It is the sole place in the codebase that must be modified when the real Hugging Face weights are wired in.

All other code depends only on the interface: `encode_dialogue(turns)` and `generate_response(prompt)`. The rest of the ESC pipeline stays unchanged when the real backbone replaces the stub.

**`encode_dialogue(turns)`** returns a dict with the structure expected by `ESCState.from_dialogue(encoder=...)`:

```python
{
  "history": Tensor[K_HISTORY_WINDOW, D_H],   # turn encodings, most recent last
  "emotion": Tensor[D_E],                      # current emotion representation
  "causes":  list of N_C dicts, each with:
               "label":     str
               "embedding": Tensor[D_C]
}
```

In the stub, all tensors are zeros. A real implementation would:
1. Tokenize the turns using the Qwen tokenizer.
2. Run a forward pass through the model.
3. Extract the last `K_HISTORY_WINDOW` hidden states for the history window, zero-padding from the left when fewer turns exist.
4. Run a lightweight emotion classifier head on the pooled representation to produce `emotion`.
5. Run a cause extraction head or use an NER/IE module to produce `causes`.

**`generate_response(prompt)`** returns a stub string. A real implementation would call `model.generate()` on a tokenized prompt that includes the conversation history and the MCTS-selected strategy as a conditioning signal.

**`encoder_adapter(backbone)`** in `data/build_states.py` wraps a `QwenBackbone`-like object into the `EncoderFn` callable type expected by `ESCState.from_dialogue()`. This adapter is the bridge between the data layer and the model layer.

---

## Shared trunk design

All networks (policy, value, transition) use the same two-layer `_SharedTrunk` template:

```
Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU
```

The default hidden width is 256. Reasons for this choice:

1. The input dimension (1283 for state-only, 1295 for state+action) is large relative to the information content. A moderate bottleneck extracts useful features without overfitting on the small dataset (approximately 2300 conversations).
2. LayerNorm before activation prevents the internal covariate shift that makes deep networks slow to train on high-variance inputs. This matters here because the state vector concatenates embeddings from very different sub-spaces (D_H = 768 history vs D_E = 128 emotion vs D_P = 3 phase).
3. ReLU rather than GELU or SiLU is used for simplicity and training stability; the networks are small enough that the approximation quality of the activation function is not the bottleneck.

---

## Parameter counts (approximate, hidden_dim=256)

| Module | Parameters |
|---|---|
| PolicyNetwork (num_actions=16) | ~460K |
| ValueNetwork | ~459K |
| LinearTransitionModel | ~596K |

These are small relative to the Qwen-9B backbone (9B parameters), reflecting the design intent: the backbone does the heavy lifting of encoding dialogue into the state representation, and the MDP-level networks are lightweight heads that learn the planning logic.
