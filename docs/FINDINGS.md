# Findings: tracing the IOI circuit with full-network attribution

This is the writeup of what the all-hidden-dim attribution view shows for GPT-2
small on the IOI task, organised around the questions Aruna raised. All numbers
are from `run_circuit.py`, `run_counterfactuals.py`, `run_paper_figures.py`, and
`run_beyond.py` (GPT-2 small, MIB `mib-bench/ioi`, 384–512 examples, RTX 4050,
each run ~10–25 s).

Method recap: **denoising attribution patching**, metric = `logit(IO) − logit(S)`,
`attr = (a_clean − a_corrupt) · ∂metric_corrupt/∂a_corrupt`. One clean forward +
one corrupt forward + one corrupt backward gives an attribution for *every*
hidden unit at once. Sanity: clean logit diff **+3.47** (paper reports 3.56),
corrupt (abc) **−0.34**; GPT-2 puts **73.7%** on the correct IO name (Fig 1).

## Q1. Does the full-network view recover the IOI circuit?

Yes. Ranking all 144 heads by total effect on the logit diff, **19 of the 26
canonical circuit heads land in the top 26** (`results/circuit_report.json`).
The per-head map (`figures/heads_total_effect.png`) and the direct-logit map
(`figures/heads_direct_logit_effect.png`, ≈ paper Fig 3b) cleanly show:

- **Name movers** 9.9 (+0.17), 9.6, 10.0 — strong positive direct effect
- **Negative name movers** 10.7 (−0.11), 11.10 (−0.08) — strong negative
- **S-inhibition** 7.3, 7.9, 8.6, 8.10 — positive, at END
- **Induction** 5.5, 6.9 — positive, at S2

The 7 heads *not* in the top-26 are exactly the ones we'd expect to be quiet:
the **backup name movers** (9.0, 10.2, 11.9 — dormant unless the main name
movers are ablated, by the paper's own account) and the **early, weak
duplicate/previous-token heads** (0.1, 0.10, 2.2, 4.11). They are not absent —
they are small under this metric (see Q4).

## Q2. What does the circuit look like, traced with this information?

`figures/circuit_diagram_annotated.png` redraws the paper's Fig 2 with our
measured per-head attribution and the information-flow arrows. The position-
resolved panels (`figures/heads_by_position.png`) recover the circuit's
*positional* structure directly from data: name movers / S-inhibition / negative
name movers fire **@END**, induction & duplicate heads fire **@S2**, previous-
token heads **@S1+1**. The class-grouped bars (`figures/heads_class_bars.png`,
≈ Fig 7) show each class's contribution and sign.

## Q3. Which paper figures did we regenerate?

| paper figure | our recreation |
|---|---|
| Fig 1 (task + GPT-2 prediction) | `figures/fig1_prediction.png` |
| Fig 2 (circuit diagram) | `figures/circuit_diagram_annotated.png` (annotated with our numbers) |
| Fig 3b (per-head direct effect on logits) | `figures/heads_direct_logit_effect.png` |
| Fig 7 (per-head contribution by class) | `figures/heads_class_bars.png` |
| (new) per-head **total** effect 12×12 | `figures/heads_total_effect.png` |
| (new) per-head effect **by position** | `figures/heads_by_position.png` |

Figures 4b / 5b (direct effect on Name-Mover *queries* / S-Inhibition *values*)
need path patching to *intermediate* nodes rather than to the logits; that is a
natural next step (swap the metric — see Caveats).

## Q4. Are there circuits *beyond* the paper? (the main question)

Three concrete "beyond the paper" results:

**(a) A consistent tail of non-canonical heads.** Heads outside the 26 that land
in the top-30 under **multiple** corruptions (so it isn't corruption noise;
`results/beyond_paper.json`):

- **9.4** — top-30 in **all four** counterfactuals, always **negative @END**:
  it behaves like an *unlisted negative name mover*.
- **11.6, 8.3, 9.3, 9.5, 11.1** — robustly important @END across 3–4 corruptions
  (more name-mover-like late heads — consistent with the paper's redundancy theme).
- **0.4, 0.5** — early heads active **@IO** under name-changing corruptions
  (candidate name/token-detection heads feeding the circuit).

So beyond the 26 there is a real, reproducible tail — mostly *additional* late
name-mover/negative-name-mover-like heads plus a couple of early name detectors.

**(b) MLP neurons.** The paper explicitly does not analyse MLPs at the neuron
level. The all-hidden-dim view ranks them directly
(`figures/top_mlp_neurons.png`): the strongest IOI neurons are
**L5.n1864, L3.n1718, L8.n1548, L6.n1636, L1.n79, L1.n824, L3.n176** — a sparse
mid-layer set worth following up.

**(c) The layer-0 MLP.** `figures/components_attn_vs_mlp.png` shows attention
carries IOI (positive L7–L9, negative L10–L11) while MLP contributions are tiny
**except the layer-0 MLP** — the one block the paper flags as necessary. We
reproduce that asymmetry from attribution alone.

**(d) Methodological: the corruption determines what you can see.**
`figures/circuit_visibility_by_corruption.png` shows that
`random_names` (which preserves the duplication structure) makes name movers most
visible but renders the **induction/duplicate-token heads nearly invisible**
(0.001 vs 0.089 under abc): a corruption that doesn't change *whether* a name is
duplicated cannot expose the duplication-detectors. This is a useful caution for
anyone reading circuits off a single counterfactual.

## Caveats

- **Attribution patching is a first-order approximation.** It is excellent for
  whole-network ranking and maps, least accurate where activations are far from
  linear (attention softmax saturation). Shortlisted nodes should be confirmed
  with exact activation/path patching.
- **The metric only sees logit-diff-relevant effects.** Previous-token and
  duplicate-token heads act through *intermediate* nodes (S-inhibition values);
  to make them pop, attribute to those nodes rather than to the logits.
- Numbers vary a few % with sample size and counterfactual; signs and rankings
  are stable.
