# nnsight notes (v0.7.0)

The brief suggested nnsight, and the repo uses it for **activation inspection**
(`src/nnsight_inspect.py`) — reading the full hidden state at every site for a
prompt with the tracing API. That path works well:

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="cuda", dispatch=True)
with model.trace(prompt):
    resid = model.transformer.h[9].output[0].save()
    mlp_hidden = model.transformer.h[9].mlp.act.output.save()
```

## Gradients: why the engine uses PyTorch hooks instead

The documented nnsight gradient idiom is:

```python
with model.trace(prompt):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    logits = model.lm_head.output
    loss = logits.sum()
    with loss.backward():
        grad = hs.grad.save()   # <-- raises on 0.7.0
```

On this machine (nnsight **0.7.0**, torch 2.11+cu128, GPT-2) this raises:

```
MissedProviderError: Execution complete but `<id>.grad` was not provided.
Did you call an Envoy out of order?
```

i.e. the per-tensor gradient hook nnsight registers never fires — the captured
output appears detached from the autograd graph at backward time. This happens
even for the exact pattern in nnsight's own docs, at any layer, for both a
`logits.sum()` loss and a scalar logit-diff loss.

Attribution patching *needs* gradients, so `src/attribution.py` captures both
activations and gradients with native PyTorch hooks
(`register_forward_hook` + per-output `register_hook`) on the same HF GPT-2 that
nnsight wraps. This is robust and version-independent, and gives identical
semantics. The smoke tests that established this are reproducible via the
scratch scripts referenced in the commit history.

If a later nnsight release fixes the `.backward()` grad capture, the engine can
be ported to nnsight by swapping the hook layer in `src/sites.py` /
`src/attribution.py`; the attribution math is unchanged.
