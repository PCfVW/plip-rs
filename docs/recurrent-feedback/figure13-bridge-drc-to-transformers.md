# Figure 13 as the Bridge: Connecting DRC Planning to Transformer Planning

**Author:** Eric Jacopin
**Date:** March 2026

---

## 1. Context

Taufeeque et al. [1, 2] showed that a Deep Repeating ConvLSTM (DRC) trained on
Sokoban develops an internal planning algorithm: path channels encode planned
moves, plan extension kernels propagate activations bidirectionally, a
Winner-Takes-All mechanism selects between competing paths, and extra recurrent
ticks refine the plan. Independently, Lindsey et al. [3] showed that Claude 3.5
Haiku commits to rhyme plans at a single token position (the "planning site")
using CLT features, and that suppress + inject interventions only affect the
model at that position.

An open-tools replication [4] reproduced Lindsey et al.'s Figure 13 in Gemma 2
2B (426K CLT) and extended the investigation to Llama 3.2 1B, identifying a
three-stage planning process — Encoding at L0 (phonological neighborhood
activation), Refinement in middle layers (narrowing the candidate set), and
Crystallization at L22-25 (final selection) [4, Section 02] — and showing that
Llama possesses the first two stages but lacks the commitment circuit that
sustains the planning signal through the final layers.

This document argues that the Figure 13 replication is the structural bridge
between the DRC and transformer planning findings, and reports the results of
two recurrent pass experiments that test the DRC analogy directly.

---

## 2. The Structural Correspondence

Both the DRC and the transformer implement the same three-phase planning
algorithm through different mechanisms:

| Phase | DRC (Sokoban) | Transformer (Rhyming) |
|-------|---------------|----------------------|
| **Initialize** | Encoder kernels detect entities (box, agent, target) and seed path channel activations near them [2, Section 5.1] | Embedding layer + early layers activate phonological features; CLT features at L0 encode rhyme-family associations [4, Section 02] |
| **Extend & Refine** | Plan extension kernels (linear + turn) propagate activations forward from boxes and backward from targets through recurrent ticks [2, Section 5.2] | Attention routes rhyme-planning signal through successive layers; mid-layer CLT features refine candidate rhyme words [4, Sections 01-02] |
| **Commit** | Path channel activations stabilize at entity squares after Winner-Takes-All competition [2, Section 5.3] | CLT features activate exclusively at the planning site (newline token); commitment circuit (L22-25 in Gemma, L14-15 in Llama) sustains signal to output [4, Sections 03-04] |

### 2.1 Specific Evidence for Each Parallel

**Representation: Path channels = CLT features.**
The DRC stores plans as activations in specific hidden-state channels. Each path
channel corresponds to a movement direction (box-up, agent-left, etc.) and
activates on the spatial grid where that movement is planned [2, Table 1]. In
the transformer, CLT features encode planned rhyme words. In Gemma 2 2B, 200
planning features show a bimodal distribution: 48.5% at L0-L1 (embedding-level
search) and 38% at L24-L25 (commitment). In Llama 3.2 1B, 82% of planning
features concentrate at L15 [4, Section 05].

**Localization: Entity squares = Planning site.**
Path channel activations localize at the squares where the box or agent will
move [2, Figure 1]. In the transformer, the Figure 13 replication shows that
planning features activate *exclusively* at the planning site (the last token
before generation). The characteristic spike — flat at ~4.5e-8 for 31 tokens,
then 0.483 at the planning site — demonstrates 10-million-fold localization [4,
Section 03, Version D]. This is the same principle: planning concentrates at the
position where the decision is made.

**Causal mechanism: Intervention scores.**
Intervening on path channels in the DRC achieves 99.7% causal intervention
score for Pooled Next Action channels [2, Table 2]. In the transformer,
suppress + inject achieves a 48% cross-group redirect (155 million x ratio) at
the planning site [4, Section 03, Version D]. Both systems have causally
efficacious planning representations, not merely predictive ones.

**Depth gradient: Ticks = Layers.**
The DRC's plan quality improves with computation: F1 score of the
Box-Directions probe increases across ticks, and the plan extends further (chain
length grows from ~3 to ~6 positive-prediction squares over 6 thinking steps)
[1, Figure 4]. In the transformer, the layer-depth gradient shows earlier
injection producing stronger planning-site effects: L16 injection (9 downstream
attention layers) gives 160,000x ratio, L19 gives 1,285x, L22 gives 774x, L23
gives 719x [4, Section 03, Version C]. The L25 case (610x) is architecturally
trivial — no downstream attention layers — but the L16-L23 gradient demonstrates
genuine planning-site routing through attention: more downstream layers = more
opportunity for the signal to accumulate at the planning position, just as more
DRC ticks = longer path channel extensions.

**Insufficient depth = Failed commitment.**
DRC(1,1) achieves lower performance than DRC(3,3), with weaker long-term
planning and more accidental cycles [1, Sections 3-4; 1, Appendix C]. Llama
3.2 1B (16 layers)
fails to commit to rhyme plans that Gemma 2 2B (26 layers) sustains: Llama's
planning features concentrate at L15 (the final layer) with no room for a
multi-layer commitment circuit, while Gemma distributes features across L0-L1
and L24-L25 with the L22-25 block functioning as a dedicated commitment circuit
[4, Sections 04-05].

### 2.2 Figure 13 as the Bridge

The Figure 13 replication provides the specific evidence that links the DRC
findings to transformer planning:

1. **Temporal localization** (Versions A-C): Planning features activate only at
   the planning site, with zero activation at all other positions. This is the
   transformer analog of path channel activations localizing at entity squares
   in the DRC.

2. **Causal effect at the planning position** (Version D): Suppress + inject
   only changes the model's output when applied at the planning site. This
   parallels the DRC's causal intervention [2, Table 2], where modifying path
   channel activations at the box's current square changes the agent's chosen
   action. In both cases, the intervention must target the specific position
   where the planning decision is made — not an arbitrary position.

3. **Layer-depth gradient** (Version C): Earlier-layer injection produces
   stronger effects through more downstream attention layers. This is the
   transformer analog of the DRC's plan extension kernels propagating
   activations through successive ticks — more computation = more propagation
   = stronger planning signal.

4. **Cross-group redirection** (Version D): Suppressing natural rhyme features
   and injecting foreign ones redirects the model's output (48% probability
   mass). This parallels the DRC's Winner-Takes-All mechanism [2, Section 5.3],
   where strengthening one path channel's activation suppresses competing paths.
   The mechanisms differ — the DRC's WTA uses inhibitory cross-channel kernels
   while the transformer's redirect uses explicit feature suppression + injection
   — but the functional outcome is the same: one plan is selected at the expense
   of others.

Without Figure 13, the connection between the DRC and transformer planning would
be only analogical ("both systems plan"). With Figure 13, the connection is
structural: both systems localize planning decisions at specific positions, both
have causally efficacious planning representations, and both require sufficient
computational depth to sustain the planning signal.

---

## 3. The Recurrent Pass Experiments

Section 2 establishes that DRC ticks and transformer layers play analogous roles
in planning. A natural question follows: if ticks help the DRC, does re-running
transformer layers help the transformer? This is not the same prediction — ticks
iterate through *gated recurrent state*, while re-running transformer layers
re-executes *stateless* attention + MLP. The experiments below test whether the
DRC's recurrence principle transfers despite this architectural difference, using
a recurrent pass through Llama's commitment layers with optional feedback
injection [4, branch anacrousis].

### 3.1 Shared Experimental Design

- **15 couplets** with known rhyme targets. Each couplet provides a first line;
  the model generates a second line; success = last word rhymes with the target.
- **Feedback vector**: for each couplet, the averaged L2-normalized embedding
  directions of all words in the target's rhyme family (e.g., for "light":
  averaged unembeddings of "light", "night", "bright", "sight", ...).
- **Planning site**: the last token position of the prompt (the newline token
  before generation begins).
- **Measurements**: generation (30 tokens, greedy, temp=0), P(target word),
  P(rhyme family), target word rank — all via softmax over the full vocabulary.
- **Baseline**: 10/15 rhyme success under standard generation (no intervention).

### 3.2 Phase 1: Prefill-Only Recurrence

**Design.** 25 conditions: baseline + 4 loop ranges x (1 double pass + 5
feedback strengths). Loop ranges: (12,15), (10,15), (8,15), (14,15). Feedback
strengths: 1.0, 2.0, 5.0, 10.0, 20.0. The recurrent block applies during
prefill only; autoregressive generation uses standard single-pass forward.

**Results.**

| Condition | Rhymes | avg P(fam) | Key observation |
|-----------|--------|-----------|-----------------|
| **Baseline** | **10/15** | 4.2e-4 | Reference |
| Double pass (no feedback), all ranges | 0--4/15 | ~4e-4 | **Catastrophic**: pure recurrence destroys rhyming |
| unembed_10-15_s=1.0 | 10/15 | 6.9e-4 | Ties baseline |
| **unembed_14-15_s=2.0** | **10/15** | **1.2e-2** | Ties baseline, **29x P(fam) increase** |
| unembed_14-15_s=5.0 | 9/15 | 4.7e-1 | P(fam) climbs but generation starts degrading |
| unembed_*_s=10--20 | 1--6/15 | 0.5--1.6 | P(fam) saturates, generation collapses |

**Interpretation.** Pure recurrence (no feedback) is destructive: all four
double-pass conditions collapse to 0--4/15. The DRC's ConvLSTM layers have gated
recurrent state (i, f, j, o gates) specifically designed for iterative refinement
[2, Appendix B, equations 5-10]. Transformer layers (attention + MLP) are
optimized for single-pass processing; re-running them amplifies noise. Guided
feedback at moderate strength (s=2.0 on L14-15) amplifies the planning
*representation* — P(rhyme family) increases 29x — but does not change the
*autoregressive trajectory*: rhyme success stays at 10/15. At higher strengths
(s=5.0+), generation quality degrades and different couplets begin to fail,
but the 5 baseline failures are never rescued by prefill-only feedback at any
quality-preserving strength.

### 3.3 Phase 2: Sustained Feedback During Generation

**Motivation.** The prefill-only experiment applied recurrence only at
initialization, then switched to standard single-pass generation. This is unlike
the DRC, where recurrence operates at every timestep — 3 ticks per environment
step [1, Section 2], plus additional "thinking steps" where the agent repeats
observations to gain extra computation time [1, Sections 3.2 and 4.1]. The prefill-only design is analogous to a DRC that
ran its ConvLSTM ticks only at the first timestep and then used a feedforward
network for all subsequent actions.

**Design.** 3 new conditions on L14-15 (the best-performing loop range from
Phase 1): sustained feedback at strengths 0.5, 1.0, 2.0. At each autoregressive
generation step, the recurrent block is applied: layers 0--13 process the
current token normally, layers 14--15 process twice with feedback injection
between passes, then the result propagates to the output. The feedback vector is
injected into the current token's residual stream (not the planning site, which
is frozen in the KV-cache).

**Results.**

| Condition | Rhymes | avg P(fam) |
|-----------|--------|-----------|
| Baseline (no intervention) | 10/15 | 4.2e-4 |
| unembed_14-15_s=1.0 (prefill only) | 9/15 | 1.4e-3 |
| unembed_14-15_s=2.0 (prefill only) | 10/15 | 1.2e-2 |
| sustained_14-15_s=0.5 | 9/15 | 7.0e-4 |
| **sustained_14-15_s=1.0** | **11/15** | **1.4e-3** |
| sustained_14-15_s=2.0 | 7/15 | 1.2e-2 |

![Strength-response curve for L14-15](figures/recurrent_feedback_strength_curve.png)

Note: P(fam) values are identical between prefill-only and sustained at the
same strength. This is expected: the logit analysis measures the model's
probability distribution after the *prefill* forward pass, which is identical
in both conditions. The difference between conditions is entirely in the
autoregressive generation trajectory.

### 3.4 Per-Couplet Analysis

The sustained_14-15_s=1.0 result (11/15) is not a statistical artifact of
different couplets trading places. It is a strict superset of the baseline
(see also per-couplet grid figure in the README):

| id | target | baseline | prefill s=1.0 | **sustained s=1.0** | prefill s=2.0 |
|----|--------|----------|--------------|---------------------|--------------|
| 1 | light | "silvered" X | "silvered" X | **"light" Y** | "silvered" X |
| 2 | play | "play" Y | "play" Y | "play" Y | "play" Y |
| 3 | sound | "sound" Y | "sound" Y | "sound" Y | "sound" Y |
| 4 | rain | "of" X | "of" X | "sky" X | "of" X |
| 5 | time | "that" X | "that" X | "that" X | "that" X |
| 6 | air | "air" Y | "air" Y | "air" Y | "air" Y |
| 7 | gold | "setting" X | "setting" X | "setting" X | "setting" X |
| 8 | fire | "fire" Y | "fire" Y | "fire" Y | "fire" Y |
| 9 | stone | "stone" Y | "stone" Y | "stone" Y | "stone" Y |
| 10 | dream | "dream" Y | "dream" Y | "dream" Y | "dream" Y |
| 11 | strange | "strange" Y | "strange" Y | "strange" Y | "strange" Y |
| 12 | love | "love" Y | "love" Y | "love" Y | "love" Y |
| 13 | truth | "girl" X | "girl" X | "girl" X | "girl" X |
| 14 | world | "world" Y | "man" X | "world" Y | "world" Y |
| 15 | earth | "earth" Y | "earth" Y | "earth" Y | "earth" Y |

**Conversion.** Couplet 1 (target: "light") is a genuine rescue. Baseline
generates "The moon is a silvered,"; sustained_s=1.0 generates "The moon is a
silver light,". The sustained feedback changed the generation trajectory to
produce the exact target word. No prefill-only condition at any strength achieved
this.

**Preservation.** All 10 baseline successes are preserved under sustained_s=1.0.
Zero regressions.

**Stabilization.** Couplet 14 (target: "world") reveals a subtlety: prefill-only
s=1.0 causes a regression (baseline "world" Y → prefill "man" X), but
sustained_s=1.0 preserves the correct output ("world" Y). The sustained
directional pressure during generation stabilizes a trajectory that the
prefill-only perturbation destabilized.

**Resistant failures.** Four couplets fail under every condition that preserves
generation quality (strengths 0.5--5.0, all loop ranges, both prefill-only and
sustained): 4 (rain), 5 (time), 7 (gold), 13 (truth). At extreme strengths
(s=10--20), where overall rhyme rate drops to 1--6/15, couplets 5, 7, and 13
occasionally succeed — but these are artifacts of the collapsed generation
regime, not evidence of planning rescue. The baseline failures are thus two
populations: one persistence failure (couplet 1, rescued by sustained feedback)
and four resistant failures (couplets 4, 5, 7, 13, unresponsive to any
quality-preserving intervention).

### 3.5 Interpretation

**Initial evidence for generation-time recurrence.** The Phase 1 result
("feedback amplifies the representation but not the output") was incomplete: it
applied recurrence only at initialization. When recurrence is sustained through
generation — matching the DRC's per-tick operation — it converts one failure
case (couplet 1) and preserves all 10 baseline successes. The sample is small
(n=15) and the improvement is one couplet, but the direction is consistent with
the DRC analogy: the key variable is **persistence of directional pressure**,
not gated recurrent state per se.

**The mechanisms differ, but the principle transfers.** In the DRC, each tick
operates on the full H x W spatial grid: the ConvLSTM can modify activations at
any grid cell, including the box and target positions [2, Section 3.1]. In the
transformer during generation, the recurrent pass can only modify the current
token's hidden state; previous tokens' representations are frozen in the
KV-cache. The planning site (the newline token from the prompt) cannot be
re-processed. Despite this constraint, the feedback works — it applies
directional pressure at the point of action (each generated token). This is
closer to the DRC's PNA (Pooled Next Action) channels — which represent the
immediate next action at every square [2, Section 4] — than to the plan
extension kernels that modify the full plan.

**Two populations of failures.** The sustained experiment discriminates between
two hypotheses about the 5 baseline failures:

- *Hypothesis (a)*: the feedback works at the planning site but the effect
  dissipates during generation. Sustained feedback should rescue these.
- *Hypothesis (b)*: no plan was ever formed; there is nothing to sustain.

Couplet 1 is hypothesis (a): sustained feedback rescues it. Couplets 4, 5, 7,
13 are consistent with hypothesis (b): they resist intervention at every
quality-preserving strength, across all loop ranges, in both prefill-only and
sustained modes. (At extreme strengths s=10--20, where generation collapses
overall, occasional successes appear for couplets 5, 7, and 13 — but these
co-occur with overall rhyme rates of 1--6/15, suggesting brute-force
perturbation rather than planning rescue.) The distinction between persistence
failures and resistant failures could not be made from the prefill-only data
alone.

**Inverted-U in feedback strength.** Sustained feedback at s=0.5 is too weak
(9/15), s=1.0 is optimal (11/15), s=2.0 is too strong (7/15 — generation
coherence collapses). The DRC shows a qualitatively similar curve: plan quality
improves with additional ticks up to a point, then levels off [1, Figure 4;
1, Section 4.1]. A caveat: any tunable intervention will exhibit an inverted-U
(too little = no effect, too much = destructive), so the shared shape is not
by itself strong evidence for a specific structural parallel. What is
informative is that the transformer's optimal strength (s=1.0) is moderate —
enough to sustain directional pressure without overwhelming the model's
generation coherence — suggesting a regime analogous to the DRC's optimal
tick count, where computation is sufficient to extend the plan but not so
excessive as to introduce noise.

---

## 4. Assessment

### 4.1 What is established

| Claim | Evidence | Status |
|-------|----------|--------|
| Transformer planning features localize at the planning site (Figure 13) | 10M x spike, 70% of suppress+inject pairs max at planning site | Done |
| CLT features = transformer analog of path channels | 200 planning features in Gemma, 82% at L15 in Llama | Done |
| Commitment circuit identified | L22-25 in Gemma (0/10 without), L14-15 in Llama (1/15 without L14) | Done |
| Insufficient depth = failed commitment | Llama 16 layers vs Gemma 26 layers; parallels DRC(1,1) vs DRC(3,3) | Done |
| Pure recurrence does not transfer from RNN to transformer | 0--4/15 for all double-pass conditions | Done |
| Guided prefill feedback amplifies planning signal | 29x P(fam) at unembed_14-15_s=2.0 | Done |
| **Sustained feedback during generation improves rhyming** | **11/15 at sustained_s=1.0 (vs. 10/15 baseline); n=15, 1 couplet converted** | **Preliminary** |
| **Persistence failure vs. resistant failure** | **Couplet 1 rescued; couplets 4, 5, 7, 13 resist quality-preserving intervention** | **Preliminary** |
| **Inverted-U in feedback strength** | **s=0.5 too weak, s=1.0 optimal, s=2.0 destructive** | **Observed** |

### 4.2 What remains

The sustained experiment tests one loop range (L14-15) at three strengths on a
single model (Llama 3.2 1B) and task (rhyme completion). Three directions would
strengthen the bridge:

1. **Gemma 2 2B replication.** The sustained feedback mechanism should work on
   L22-25 in Gemma, where the commitment circuit is stronger and the CLT
   features are more granular. Different model, same predicted outcome.

2. **Larger couplet set.** The 15-couplet probe set is small. A larger set would
   test whether the persistence/resistant failure distinction holds at scale and
   provide statistical power for the rhyme-rate improvement (currently n=15,
   one couplet converted).

3. **Cross-task transfer.** The structural correspondence (Section 2) predicts
   that sustained directional pressure should help in any planning task where
   the model's forward pass provides insufficient depth. Acrostic poetry
   (planning the first letter of each line) and text-based Sokoban (planning
   a move sequence) are natural test cases [4, planning-tasks-report].

---

## 5. Conclusions

### 5.1 Figure 13 is the bridge

The Figure 13 replication provides five specific structural parallels between the
DRC and the transformer (Section 2.2): temporal localization, causal efficacy at
the planning position, layer-depth gradient, cross-group redirection, and
insufficient-depth failure. Without Figure 13, the connection is merely
analogical. With it, the same planning algorithm — Initialize, Extend & Refine,
Commit — is visible in both architectures through mechanistic evidence.

### 5.2 The planning algorithm is shared; the refinement mechanism is architecture-specific

Both the DRC and the transformer learn to localize planning at specific positions,
encode plans in specific features, and require sufficient computational depth.
But the *refinement* step operates through fundamentally different mechanisms:
the DRC uses gated recurrent state (ConvLSTM ticks), while the transformer uses
depth (successive layers). Pure recurrence — re-running transformer layers
without guidance — is destructive (0--4/15), because transformer layers lack the
gated state that allows incremental refinement.

### 5.3 Sustained directional pressure is the partial bridge

When feedback is sustained through generation (the transformer analog of the
DRC's per-tick recurrence), it converts one failure (11/15 vs. 10/15 baseline)
and preserves all successes. The improvement is small (n=15, one couplet
converted), and replication on larger datasets and other models is needed before
drawing strong conclusions. But the direction is consistent with the DRC
analogy: the plan must be maintained throughout execution, not just computed at
initialization. The mechanisms differ — ConvLSTM state vs. repeated vector
injection into the residual stream — but the functional principle transfers.

The couplet 1 conversion is particularly clean: baseline generates "silvered"
(off-topic); sustained feedback generates "light" (exact target). No prefill-only
condition at any strength achieved this. The plan existed in the model's
representations but dissipated during generation — sustained pressure prevented
the dissipation.

### 5.4 Two populations of planning failures

The sustained experiment reveals that what appeared to be 5 homogeneous failures
under prefill-only conditions are in fact two distinguishable populations:

- **Persistence failures** (1/5): the model forms a plan at the planning site
  but cannot sustain it through generation. Sustained directional pressure
  rescues these. Couplet 1 ("light") is this type.

- **Resistant failures** (4/5): the model does not respond to any
  quality-preserving intervention. Couplets 4 (rain), 5 (time), 7 (gold),
  13 (truth) are this type. At extreme feedback strengths (s=10--20) that
  collapse overall generation quality, couplets 5, 7, and 13 occasionally
  succeed, but only couplet 4 fails under every condition tested.

The specific mechanism behind the resistant failures is not established by the
current experiment. Possible causes include: (a) absent plan initialization —
the model's phonological encoding does not seed a rhyme plan for these couplets;
(b) semantic incompatibility — the prompt context constrains the generation
trajectory too strongly for the rhyme signal to redirect it; (c) feedback vector
misalignment — the averaged unembedding direction does not capture the correct
planning representation for these rhyme families. Disambiguating these
hypotheses would require CLT feature analysis at the planning site for each
failing couplet.

This distinction has implications for the DRC parallel. The DRC also exhibits
planning failures on hard levels — but its failures are predominantly
depth-limited (more ticks help) [1, Sections 3-4]. The transformer's resistant
failures suggest a qualitatively different failure mode that is not addressed by
additional computation or directional pressure alone.

---

## 6. Summary

The Figure 13 replication establishes a structural bridge between DRC planning
(Taufeeque et al.) and transformer planning (Lindsey et al.):

- Both architectures localize planning at specific positions (entity squares /
  planning site).
- Both encode plans in specific features (path channels / CLT features).
- Both require sufficient computational depth (ticks / layers).
- Both have causally efficacious planning representations.

Two recurrent pass experiments tested the DRC analogy directly:

- **Prefill-only recurrence** (Phase 1): pure recurrence is destructive;
  guided feedback amplifies the planning signal 29x but does not change
  the generation trajectory.
- **Sustained recurrence** (Phase 2): feedback applied at every generation
  step achieves 11/15 (n=15, one couplet converted, zero regressions). This
  provides initial evidence that the DRC's per-tick recurrence has a
  transformer analog: sustained directional pressure during generation.

The planning *algorithm* — Initialize, Extend & Refine, Commit — is shared
across both architectures studied. The *refinement mechanism* is
architecture-specific (gated recurrent state vs. depth). Sustained directional
pressure is the partial bridge: it transfers the functional principle of
per-tick recurrence to the transformer, despite the architectural differences.
Replication on larger datasets and additional models is needed to determine
whether these findings generalize.

---

## References

1. M. Taufeeque, P. Quirke, M. Li, C. Cundy, A. D. Tucker, A. Gleave, and
   A. Garriga-Alonso, "Planning in a recurrent neural network that plays
   Sokoban," *arXiv:2407.15421*, 2024.

2. M. Taufeeque, A. D. Tucker, A. Gleave, and A. Garriga-Alonso,
   "Path Channels and Plan Extension Kernels: a Mechanistic Description of
   Planning in a Sokoban RNN," *NeurIPS 2025 Workshop: Mechanistic
   Interpretability*, arXiv:2506.10138, 2025.

3. J. Lindsey et al., "On the Biology of a Large Language Model," Anthropic,
   Transformer Circuits Thread, 2025.

4. E. Jacopin, "Replicating 'Planning in Poems' with Open Tools" (PLIP-rs,
   branches: melometis, tragos, anacrousis).
