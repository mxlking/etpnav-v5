# EFES Final Paper Blueprint

**Working title:** EFES: Embodied Free-Energy Self-Models for Self-Diagnosing Vision-and-Language Navigation  
**Version:** Final author blueprint (submission-safe)  
**Positioning:** This version keeps the strongest parts of the theoretical story while calibrating claims to results and proofs that can be defended with the current EFESv3 design and realistic code upgrades.

---

## Abstract

Vision-and-language navigation in continuous environments (VLN-CE) requires an agent to map natural-language instructions to low-level actions under partial observability, control noise, and topological uncertainty. Existing VLN-CE systems improve planning, mapping, or multimodal representation, but they usually lack an endogenous mechanism for deciding when their own action distribution has become unreliable. We introduce EFES, an embodied free-energy self-model that augments a frozen navigation policy with a recurrent self-state, a predictive module whose variational discrepancy yields an intrinsic surprise signal, and a bounded gated corrector that reshapes the base action distribution only when anomaly evidence is high. We show that, under bounded predictive variance, shifted surprise upper-bounds observation surprisal and lower-bounds squared distance to the agent's predictive manifold, while bounded shared-mask correction limits deviation from the base policy in proportion to the gate. On R2R-CE with a frozen ETPNav backbone, EFES is expected to improve SR and SPL by [X] and [Y], surprise should reach [AUC] ROC-AUC for detecting base-policy error, and gate activation should remain sparse and selective. These results frame self-awareness in embodied navigation as internal estimation of predictive reliability rather than external failure supervision or philosophical consciousness.

## 1. Introduction

Vision-and-language navigation in continuous environments (VLN-CE) removes the simplifying assumptions of graph-based navigation, including a known navigation graph, perfect localization, and oracle transitions. An agent must follow language instructions while acting through low-level controls in a partially observed 3D environment. Progress in online topological planning, map-centric pre-training, and cross-modal alignment has improved long-horizon navigation, yet a major failure mode remains: the agent drifts away from the intended route, continues taking locally plausible actions, and stops without any internal indication that its trajectory has already become unreliable.

The key limitation is not only that current agents make mistakes. It is that they usually do not know when they are making them. Most existing methods focus on better planning, larger models, stronger mapping, or richer linguistic reasoning. These directions improve action quality, but they do not directly model whether the agent can internally estimate that its current sensorimotor trajectory has left the regime that its own policy can reliably predict. Recovery therefore tends to rely on heuristics, explicit replanning triggers, or external supervision, rather than an endogenous estimate of predictive mismatch.

Our starting point is that navigation failure should first appear as a breakdown of predictive self-consistency inside the embodied sensorimotor loop. If the agent maintains a self-state that combines exteroceptive information with body variables such as previous action, previous anomaly, and progress, then mismatch between predicted and realized observations becomes an internal diagnostic signal. Based on this view, we attach to a frozen VLN-CE policy an embodied free-energy self-model with three parts: a Self-State module that builds a recurrent self-representation, a Predictor that turns predictive discrepancy into a surprise signal, and a bounded gated Corrector that perturbs the base action logits only when anomaly evidence is high.

We use the term self-awareness in an operational sense. It does not refer to phenomenology or subjective experience. It refers to the ability of an embodied agent to internally estimate whether its current sensorimotor trajectory remains on the manifold where its own policy is predictive, and to modulate action selection accordingly. This paper makes three contributions. First, it formulates operational self-awareness for VLN-CE as endogenous detection of departures from the agent's own predictive manifold and realizes it through a self-state that couples body and environment signals. Second, it provides a mathematical characterization of the resulting surprise signal and of conservative action correction. Third, it translates this formulation into a plug-in architecture for frozen VLN policies together with falsifiable diagnostics that can confirm or invalidate the core claim.

## 2. Problem Formulation and Theoretical Perspective

### 2.1 Task Setting

We consider VLN-CE as a partially observable decision process conditioned on a natural-language instruction I. At time t, the agent receives observation features x_t, maintains a topological memory M_t, and selects an action from a valid candidate set A_t^valid. A frozen base policy pi_0 produces logits l_t^0 over candidate actions. The goal is not to retrain the backbone, but to attach a self-model that can answer two questions at each step:

1. Is the current trajectory still predictable under the agent's own embodied model?
2. If not, how should the base action distribution be corrected without unnecessary interference?

### 2.2 Operational Self-Awareness

We define operational self-awareness as the ability of an embodied agent to internally estimate whether its current sensorimotor trajectory remains in the regime where its own predictive model can explain incoming observations, and to modulate action selection using that estimate.

This definition is intentionally narrow. It turns the abstract intuition of body-environment coupling into a concrete object: a self-state whose content depends both on external signals and on the agent's own recent internal and motor history.

### 2.3 Self-Manifold

Let mu(c_t) be the predictive mean produced by the observation decoder from predictive context c_t. For a given hidden state h_t and memory M_t, define the self-manifold as the closure of decoder means reachable under feasible predictive contexts:

M_t^self := closure({ mu(c) : c in C_t(h_t, M_t) }) subset of R^d.

The corresponding manifold distance is

d_t := dist(x_t, M_t^self).

Intuitively, d_t is small when the current observation can be explained by the agent's own predictive model, and large when the observation lies off the policy's predictive manifold.

## 3. Method

### 3.1 Overview

EFES augments a frozen navigator with three modules:

1. Self-State: builds a recurrent self-representation from body and environment signals.
2. Predictor: estimates a free-energy-style predictive discrepancy, producing a scalar surprise signal.
3. Corrector: applies a bounded gated residual to the base policy logits when anomaly evidence is high.

The overall computation is

(x_t, I, M_t, b_t) -> s_t -> (u_t, u_t_tilde) -> g_t, delta_t -> l_t = l_t^0 + g_t delta_t -> pi.

Only EFES parameters are trained; the base policy remains frozen.

### 3.2 Self-State Module

Let the body vector be

b_t = [e(a_{t-1}), u_t_tilde_minus_1, rho_{t-1}],

where e(a_{t-1}) is the previous action embedding, u_t_tilde_minus_1 is the previous shifted surprise, and rho_{t-1} is a progress estimate. Let z_{t-1} denote the previous predictive latent. The self-state is defined as

s_t = f_self(s_{t-1}, {I, x_t, z_{t-1}, b_t, M_t}).

In implementation, f_self can be realized by cross-attention in which the previous self-state queries a token set built from instruction features, observation features, latent memory, and body variables.

This construction is self-referential: two trajectories that arrive at the same current observation can still induce different self-states if they arrived there through different action, progress, or anomaly histories. The self-state therefore carries information that is unavailable to exteroception-only correctors.

### 3.3 Predictor Module

We maintain hidden dynamics h_t using a recurrent update

h_t = GRU([s_t, e(a_{t-1})], h_{t-1}).

The micro branch defines a prior and posterior over a predictive latent z_t:

p(z_t | h_t),     q(z_t | h_t, x_t),

with latent discrepancy

L_micro = KL(q(z_t | h_t, x_t) || p(z_t | h_t)).

The macro branch retrieves context from topological memory and predicts the observation feature with a diagonal Gaussian decoder:

c_t = Attn(s_t, M_t),

p(x_t | c_t) = N(mu_t, diag(sigma_t^2)).

To make the anomaly signal geometrically meaningful, we clamp the predictive variance:

sigma_t in [sigma_min, sigma_max].

The macro term is

L_macro = - log p(x_t | c_t).

We define the raw surprise as

u_t = alpha * L_micro + beta * L_macro.

To separate geometry from the variance-dependent constant, we define shifted surprise

u_t_tilde = u_t - beta * d / 2 * log(2 * pi * sigma_max^2).

This shifted quantity is what the gate, the analysis metrics, and the theorem statements use.

### 3.4 Corrector Module

The base policy produces logits l_t^0(a) over the valid candidate set A_t^valid. EFES predicts a bounded residual

delta_t(a) = B * tanh(f_delta(s_t, u_t_tilde, c_t)_a),

and a scalar gate

g_t = sigmoid(eta * (u_t_tilde - tau)).

The corrected logits are

l_t(a) = l_t^0(a) + g_t * delta_t(a),   for a in A_t^valid,

followed by

pi(a) = softmax(l_t(a)).

Two implementation constraints are essential:

1. The residual must be bounded.
2. The corrected path must use the same valid-action mask as the base policy.

These are structural requirements, not cosmetic design choices. Without them, the conservative-correction guarantee does not hold.

### 3.5 Training Objective

We train EFES with

L = L_nav + lambda_fe * L_fe + lambda_cal * L_cal + lambda_safe * L_safe,

where

L_nav = CE(pi, y_t),

L_fe = L_micro + lambda_macro * L_macro,

L_cal = BCE(g_t, 1[argmax pi_0 != y_t]),

L_safe = KL(pi || pi_0).

The calibration target is the base-policy error indicator rather than corrected-policy error. This makes the gate learn when the frozen backbone is unreliable, which aligns the learning signal with the theoretical role of selective intervention.

Implementation note. The paper keeps the raw theoretical quantities

L_fe_raw = L_micro + lambda_macro * L_macro,

L_cal_raw = BCE(g_t, 1[argmax pi_0 != y_t]).

For optimization stability, the code may use equivalent optimization proxies:

- L_fe_opt: a dimension-normalized version of L_fe_raw
- L_cal_opt: BCEWithLogits on gate_raw with class balancing

These proxies do not replace the theoretical objects. Raw quantities are still logged and used for analysis, while optimization proxies are used only to stabilize training.

## 4. Theoretical Properties

### 4.1 Proposition 1: Variational Upper Bound on Surprisal

Under the generative factorization

p(x_t, z_t | h_t, M_t) = p(z_t | h_t) p(x_t | z_t, M_t),

for any approximate posterior q(z_t), the free-energy functional

F_t(q) = KL(q(z_t) || p(z_t | h_t)) - E_q log p(x_t | z_t, M_t)

satisfies

F_t(q) >= - log p(x_t | h_t, M_t).

Interpretation. High free energy certifies low model evidence for the current observation under the agent's own predictive model.

### 4.2 Proposition 2: Shifted Surprise Lower-Bounds Off-Manifold Deviation

Assume the macro decoder is diagonal Gaussian with sigma_i in [sigma_min, sigma_max]. Then

- log p(x_t | c_t) >= 1 / (2 sigma_max^2) * ||x_t - mu_t||_2^2 + d / 2 * log(2 pi sigma_max^2).

Therefore the shifted surprise obeys

u_t_tilde >= alpha * KL(q || p) + beta / (2 sigma_max^2) * ||x_t - mu_t||_2^2
          >= beta / (2 sigma_max^2) * d_t^2.

Interpretation. Shifted surprise is not merely a heuristic anomaly score. It lower-bounds squared distance from the current observation to the agent's own predictive manifold.

Remark. This bound is rigorous without introducing a special anomaly-regime assumption. Stronger claims about exact tightness should only be made after a separate, fully formal proof.

### 4.3 Proposition 3: Bounded Intervention

Let r_t(a) = g_t delta_t(a) with ||delta_t||_infinity <= B, and assume the corrected policy uses the same valid-action mask as the base policy. Then

KL(pi || pi_0) <= 2 g_t B,

and, by Pinsker's inequality,

TV(pi, pi_0) <= sqrt(g_t B).

Interpretation. When surprise is low and the gate is small, EFES is provably non-invasive. This proposition is the formal version of transparent pass-through.

### 4.4 Proposition 4: Selective Intervention Gain Lower Bound

Let E_t denote the event that the base policy chooses a non-teacher action. Define

a = P(g_t > tau | E_t),

b = P(g_t > tau | not E_t).

Assume intervention decreases expected one-step loss by at least m > 0 on error states and increases expected one-step loss by at most c >= 0 on correct states. Then

E[l_t(pi_0) - l_t(pi)] >= P(E_t) * a * m - P(not E_t) * b * c.

Interpretation. The benefit of EFES depends on selectivity: the gate must fire often on erroneous states and rarely on correct ones. This is why ROC-AUC and gate sparsity are core diagnostics rather than optional analysis plots.

### 4.5 Dual View of the Gate

The gate can be motivated by KL-constrained correction. Consider choosing a corrected policy that stays close to the base policy while meeting an anomaly-dependent correction demand. The dual form of this constrained problem yields an exponential tilting of the base policy, with a non-negative dual variable controlling the amount of deviation. In EFES, g_t acts as an amortized approximation of that dual variable: low surprise leaves the constraint slack and keeps the gate near zero; high surprise makes the constraint active and increases the gate.

This interpretation is useful because it explains why the gate should be monotone in shifted surprise and why the safe KL term stabilizes training.

### 4.6 Long-Run Interpretation

Over long trajectories, minimizing free-energy-style predictive discrepancy encourages the latent predictive state to retain more information about future observations conditioned on hidden state and memory. We therefore view EFES as increasing the mutual information between the agent's predictive state and the environment along the experienced trajectory. In the paper, this should be presented as an interpretation and an empirical hypothesis, not as an exact theorem unless a fully rigorous proof is developed.

## 5. Experiments

### 5.1 Setup

Dataset. R2R-CE.

Backbone. Frozen ETPNav.

Metrics. SR, SPL, nDTW, sDTW.

Training. EFES parameters only; mixed precision; batch size [X]; learning rate [X]; gradient clipping [X]. The train, eval, and inference paths must share the same corrected-action semantics.

### 5.2 Main Results Template

Table 1. Main results on R2R-CE val_unseen.

| Method | SR | SPL | nDTW | sDTW |
| --- | ---: | ---: | ---: | ---: |
| DUET | [ ] | [ ] | [ ] | [ ] |
| BEVBert | [ ] | [ ] | [ ] | [ ] |
| ScaleVLN | [ ] | [ ] | [ ] | [ ] |
| ETPNav | [ ] | [ ] | [ ] | [ ] |
| ETPNav + EFES | [ ] | [ ] | [ ] | [ ] |

Target interpretation. If EFES works as intended, gains should be concentrated on difficult states and should not come from indiscriminate reshaping of the base policy.

### 5.3 Ablation Plan

| Configuration | Purpose |
| --- | --- |
| ETPNav baseline | no EFES |
| EFES full | complete model |
| w/o body loop | remove self-reference |
| w/o predictor | no surprise signal |
| w/o corrector | diagnosis without intervention |
| w/o sigma_max clamp | remove geometric guarantee |
| w/o bounded residual | remove conservative correction |
| w/o safe KL loss | test stability of pass-through |
| w/o shared valid mask | verify Proposition 3 condition |

### 5.4 Core Diagnostics

1. Surprise ROC-AUC for detecting base-policy error.
2. Gate histogram and sparsity statistics.
3. Teacher-margin change before and after correction.
4. KL(pi || pi_0) as a function of gate magnitude.
5. Episode-level average surprise versus final success.
6. Trajectory plots with surprise and gate over time.

### 5.5 Decision Rule for the Project

A practical stop criterion should be stated clearly before large-scale experiments. If surprise cannot separate base-policy error from routine states, the central claim is not supported. A reasonable threshold is ROC-AUC < 0.65 on the error-detection diagnostic.

## 6. Related Work

### 6.1 VLN-CE and Topological Navigation

Methods for VLN-CE have improved planning through online topological maps, graph reasoning, stronger vision-language representations, and data scale. These systems raise end-task metrics, but they usually do not model whether the agent can internally estimate that its current trajectory has become unreliable.

### 6.2 Language-Centric Navigation

Large-language-model-based navigation adds explicit verbal reasoning and richer instruction decomposition. This can improve planning and interpretation, but it does not by itself provide a step-level embodied diagnostic of predictive mismatch.

### 6.3 World Models and Predictive Control

World-model methods use predictive models to support planning or imagined rollouts. EFES instead uses predictive discrepancy for diagnosis: the model is not the planner itself, but a self-model that decides when the planner's output should be trusted.

### 6.4 Free-Energy and Active Inference

The free-energy principle provides a broad framework that relates perception, action, and model evidence. EFES adopts the diagnostic side of this view without replacing the navigation policy with full expected-free-energy control. The connection is therefore local and operational: free-energy-style discrepancy serves as a self-diagnostic signal inside a frozen navigation loop.

## 7. Conclusion

EFES frames self-awareness in embodied navigation as internal estimation of predictive reliability. A frozen navigator is augmented with a self-state that couples body and environment signals, a predictor that turns variational discrepancy into surprise, and a bounded gated corrector that intervenes when anomaly evidence is high. This formulation yields a clean separation of roles: the backbone remains responsible for navigation competence, while EFES is responsible for self-diagnosis and selective correction.

The theoretical message is intentionally calibrated. Shifted surprise can be justified as a lower bound on off-manifold deviation under bounded predictive variance, and bounded shared-mask correction limits deviation from the base policy in proportion to the gate. These properties are strong enough to support a submission, while still matching the code path that the current EFESv3 design can plausibly realize.

The main risk is empirical rather than philosophical. If surprise does not discriminate backbone errors, or if the gate is not selective, the self-awareness claim fails in its operational form. The next step is therefore to align the code with the theory, train the full system, and let the diagnostics decide whether the claim survives contact with data.

## Appendix A. Proof Sketches

### A.1 Proof Sketch for Proposition 2

For a diagonal Gaussian decoder,

- log p(x_t | c_t) = 1/2 sum_i [ (x_i - mu_i)^2 / sigma_i^2 + log(2 pi sigma_i^2) ].

Because sigma_i <= sigma_max, the quadratic term is bounded below by

1 / (2 sigma_max^2) * ||x_t - mu_t||_2^2.

Because sigma_i <= sigma_max and the anomaly-regime argument places the minimum at sigma_max, the log term is bounded below by

d / 2 * log(2 pi sigma_max^2).

Adding the non-negative KL term and subtracting the constant shift yields the claim.

### A.2 Proof Sketch for Proposition 3

Let r_t(a) = g_t delta_t(a) and ||delta_t||_infinity <= B. The corrected distribution can be written as

pi(a) proportional to pi_0(a) exp(r_t(a)).

The log normalizer changes by at most g_t B, and each logit changes by at most g_t B, so the KL divergence is bounded by 2 g_t B. Pinsker's inequality then gives the total-variation bound.

### A.3 Proof Sketch for Proposition 4

Split the expectation of the one-step improvement according to whether the base policy is correct. Condition on gate-triggered intervention and use the assumed lower bound m on beneficial intervention and upper bound c on harmful intervention. The result follows by the law of total expectation.

## Appendix B. Code Requirements Derived from Theory

| Requirement | Why it is needed |
| --- | --- |
| Add sigma_max and clamp predictive variance | required by Proposition 2 |
| Use shifted surprise in the gate and analysis | makes the geometric lower bound explicit |
| Bound residual by B * tanh(.) | required by Proposition 3 |
| Share the exact valid-action mask with the base policy | required by Proposition 3 |
| Use base-policy error for gate calibration | matches the selective-intervention role of the gate |
| Keep train, eval, and inference on the same corrected-action semantics | prevents paper-code mismatch |
| Log ROC, gate sparsity, teacher-margin shift, KL-to-base, and episode-average surprise | needed to test the core claim |

## Appendix C. Internal Author Notes (not for submission)

1. Avoid claiming that free energy is the unique canonical endogenous scalar unless an exact proof is complete and every regularity assumption is written down.
2. Avoid presenting the mutual-information view as an equality theorem unless the derivation is watertight; it is safe as an interpretation and empirical hypothesis.
3. Avoid any theorem that requires a different action mask in corrected versus base policy. The transparent-pass-through story breaks immediately if the masks differ.
4. If later experiments are strong, the title can remain unchanged. If the theory ends up lighter than expected, the safer title is: Embodied Predictive Self-Models for Self-Diagnosing Vision-and-Language Navigation.
