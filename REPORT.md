# Dual-Route Induction — Experiment Report

## Experiment Status (as of 2026-04-08)

| Experiment | 8B | 70B |
|---|---|---|
| Causal scores (TokCopy/ConCopy) | ✅ | 🔄 in progress (4 shards, not merged) |
| Head rankings (token/concept) | ✅ | ⏳ pending causal scores |
| Attention scores (NTM/LTM, concept entities) | ✅ | ✅ |
| Attention scores (NTM/LTM, random tokens) | ✅ | ❌ not run |
| Improbable bigram traces + scores + DLA | ✅ | ✅ |
| Random token traces + scores + DLA | ✅ | ❌ not run |
| Concept traces + scores + DLA | ✅ | ✅ |
| Three-condition LTM/NTM plot | ✅ | ⏳ needs causal scores |
| Three-condition DLA plot | ✅ | ⏳ needs causal scores |
| Mean activations (ablation baseline) | ✅ | ❌ not run |
| Mean-ablation interventions | ✅ | ⏳ needs causal scores + mean activations |

Everything remaining for 70B is blocked on merging the causal score shards.

---

## Key Findings — Llama-3.1-8B

**Dataset**: 100 improbable bigrams; 33/100 hallucinate on second token (correct prefix, wrong suffix).

### LTM/NTM across three conditions

- **Token heads show suppressed NTM on improbable bigrams** vs random tokens and concepts. E.g. head (15,28): NTM = 0.31 (improbable) vs 0.49 (random) vs 0.35 (concepts). Consistent across nearly all top token heads — improbable tokenization disrupts their in-context copying.
- **Concept head LTM is inflated for improbable bigrams** in some heads. Head (15,16): LTM = 0.42 (improbable) vs 0.26 (random) vs 0.13 (concepts) — abnormally attending to the last token for improbable inputs. Head (20,1) also elevated (0.14 vs 0.025 random).
- Head (13,27) shows consistently high LTM across all three conditions (~0.08–0.12) — general last-token copier, not condition-specific.

### DLA analysis

- Most token heads still produce **positive correct-token DLA** on improbable bigrams (e.g. (16,20): +0.11, (30,24): +0.11) — weakly promote the right answer despite suppressed NTM.
- Strongest correct-token DLA from **concept heads (20,1) (+0.25) and (27,20) (+0.17)** — concept heads are the primary promoters of the correct suffix on improbable bigrams.
- Head **(13,27)**: negative correct DLA (−0.04) and positive wrong-token DLA (+0.04) on improbable bigrams — active hallucination driver.
- Head **(31,14)**: DLA = −0.27 (correct) and −0.57 (wrong) — suppresses all token predictions.

### Mean-ablation results (`cache/ablation/Llama-3.1-8B/summary_start0_stopend.json`)

| Condition | Copy% | Hall% | PrefFail% |
|---|---|---|---|
| Baseline | 65% | 35% | 0% |
| Top-1 concept ablated | 69% | 31% | 0% |
| Top-8 concept ablated | 67% | 32% | 1% |
| Top-16 concept ablated | 67% | 30% | 3% |
| Top-1 token ablated | 36% | 64% | 0% |
| Top-2 token ablated | 41% | 59% | 0% |
| Top-4 token ablated | 40% | 60% | 0% |
| Top-8 token ablated | 6% | 94% | 0% |
| Top-16 token ablated | 0% | 47% | 53% |

Note: at top-16 token ablated, hallucination appears to drop from 94% → 47% because the model now also fails on the **prefix** for 53% of examples (copy% + hall% + prefix-fail% = 100%).

**Key conclusion**: Token induction heads are load-bearing for correct suffix copying. Concept head ablation has only a minor protective effect (35% → 30% hallucination). Token head ablation is far more causally potent but destroys copying capacity entirely.

### Primary intervention targets

- Head **(13,27)**: top-ranked concept head, positive wrong-token DLA, persistently high LTM regardless of condition.
- Head **(15,16)**: anomalously inflated LTM on improbable bigrams specifically.

---

## Correlation with Paper Hypotheses

### Hypothesis 1 — Token heads work functionally during hallucinations

Paper predicts: NTM consistent across conditions, positive correct-token DLA.

**Result: Possibility 1-1 confirmed.** NTM is *suppressed* on improbable bigrams for nearly all top token heads. The signal is degraded in attention routing, not in output projection. Ablation curve is the strongest evidence: 1 token head ablated → hallucination doubles (35% → 64%); 8 ablated → 94%. Token heads are more load-bearing than the paper's framing anticipated.

### Hypothesis 2 — Concept heads fail during hallucinations

Paper predicts: Possibility 2-1 — elevated LTM on improbable bigrams (concept heads treating bigrams as semantic concepts).

**Result: Confirmed, but causal weight is smaller than implied.** Head (13,27): persistently high LTM, negative correct-token DLA, positive wrong-token DLA. Abstract claims "heads contributing to incorrect predictions are identical to concept induction heads" — this holds for DLA but not for causal ablation. Concept head ablation reduces hallucination by only ~5pp even at top-16. DLA implication ≠ causal responsibility.

### Intervention — Mean-ablation localizes and corrects hallucinations

Paper predicts: ablating faulty concept heads prevents hallucinations.

**Result: Directionally correct but weak — closer to Possibility 3-2.** Concept head ablation gives a modest protective effect. Token head ablation is far more causally potent but catastrophically worsens performance. The third contribution ("causal interventions promote correct copying") holds only weakly for concept heads.

### Revised narrative

The data supports the dual-route framing but shifts emphasis. The story is not "concept heads misfire and cause hallucinations" but rather **"improbable bigrams break token heads, and concept heads are an insufficient fallback that sometimes makes things worse."** The two populations are near-orthogonal (only 1 shared head in top-15 of each), confirming the dual-route structure. The abstract's claim about concept heads being the primary failure locus should be softened — they are secondary contributors whose role is implicated by DLA but not confirmed by causal intervention.
