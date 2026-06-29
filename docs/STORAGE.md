# The storage problem

Looking at *every hidden dimension across the whole network* means the raw
quantity of numbers is large. This note explains the accounting and the choices
this repo makes.

## How big is "all hidden activations"?

For GPT-2 small we track, per layer, five components:

| component   | width |
|-------------|------:|
| resid_pre   |   768 |
| attn_out    |   768 |
| mlp_hidden  |  3072 |
| mlp_out     |   768 |
| resid_post  |   768 |
| **per-layer total** | **6144** |

Across 12 layers that is **73,728 hidden dimensions per token position**.

The *full* attribution tensor has shape

    [n_examples, n_sites, seq_len, dim]

For 1,000 examples at seq_len ≈ 19 that is on the order of

    1000 × 73,728 × 19 ≈ 1.4 billion floats ≈ 5.6 GB (fp32)

per direction — and that is just GPT-2 small. This is the thing that does not
fit comfortably if you keep it naively.

## The trick: reduce as you stream

Attribution patching gives a value for *every element*, but you almost never
need every element at once. `AttrAccumulator` (`src/attribution.py`) keeps three
running reductions and updates them batch-by-batch, so peak memory is
`O(n_sites × seq × dim)` — **independent of `n_examples`**:

1. **`scalar`** — sum over (position, dim): one number per site.
   → the network-level circuit map. ~60 floats.
2. **`per_dim`** — sum over (position), kept per hidden dimension.
   → the "all hidden dims" view. 73,728 floats ≈ 0.3 MB compressed.
3. **`per_pos`** — sum over (dim), end-aligned to the last K positions.
   → where in the sentence the signal lives. ~480 floats.

A 64-example run writes **`per_dim.npz` ≈ 0.3 MB** instead of gigabytes.

## On-disk layout (`results/`)

| file            | content                                   | size order |
|-----------------|-------------------------------------------|-----------:|
| `per_dim.npz`   | per-hidden-dim attribution, one array/site | ~MB |
| `network.csv`   | one row per (layer, component)             | ~KB |
| `positions.csv` | one row per (site, position-from-end)      | ~KB |
| `manifest.json` | run metadata + storage accounting          | ~KB |
| `summary.json`  | top sites / top units                      | ~KB |

## If you *do* want element-level data

Options, in increasing heaviness:
- **Top-k only**: keep the k largest-|attr| units per site (sparse).
- **Chunked memmap / Zarr**: stream `[examples × seq × dim]` per site to a
  chunked, compressed array on disk; read slices lazily.
- **`safetensors` shards**: one file per (site) or per (layer), fp16.

The streaming-reduction approach here is deliberately the lightweight default;
the hooks in `src/sites.py` give you the raw per-element tensors if you want to
swap in any of the above.
