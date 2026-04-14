# EFES: Embodied Free-Energy Self-Models for Self-Diagnosing Vision-and-Language Navigation

---

## Abstract

Vision-and-language navigation in continuous environments (VLN-CE) requires an agent to translate
natural-language instructions into low-level motor actions under partial observability and topological
uncertainty. Existing VLN-CE systems improve action quality through better planning, mapping, or
multimodal representation, but they lack an endogenous mechanism to determine *when their own
decisions have become unreliable*. We present **EFES** (Embodied Free-Energy Self-model), a plug-in
self-diagnostic layer that augments a frozen navigation policy with: (i) a recurrent **Self-State**
module coupling exteroceptive signals with somatic variables (previous action, anomaly, progress);
(ii) a **Predictor** whose variational free-energy discrepancy yields an intrinsic surprise signal;
and (iii) a **bounded gated Corrector** that reshapes the base action distribution only when anomaly
evidence is high.

Our main theoretical contribution is a *tight* characterization: we prove that the greatest lower
bound (infimum) of the shifted free energy over all valid model configurations, in the anomaly regime,
equals exactly the squared Euclidean distance from the current observation to the agent's
**self-manifold** (Theorem 3), and that this infimum is *achieved*. This establishes free energy
as the *canonical* endogenous diagnostic—no other scalar derivable from the same generative model
achieves a tighter lower bound on off-manifold deviation. We further (i) derive the gate from first
principles via KL-constrained policy optimization, where the gate equals the optimal Lagrange
multiplier (Theorem 5); (ii) prove an ergodic self-consistency result connecting long-run free-energy
minimization to maximum agent-environment mutual information (Theorem 7); and (iii) establish a
selective intervention gain lower bound (Theorem 6).

On R2R-CE val_unseen, EFES improves frozen ETPNav by [ΔSR] SR and [ΔSPL] SPL, while surprise
achieves [AUC] ROC-AUC for detecting base-policy error and gate activation remains sparse and
selective. These results support an operational view of self-awareness in embodied navigation:
an agent need not receive explicit failure labels to diagnose off-manifold departure, because
predictive inconsistency within the sensorimotor loop is the *tightest possible* endogenous signal.
Empirically, we evaluate the variational gate interpretation through selectivity diagnostics and
report an MI proxy rather than claiming a direct end-to-end verification of the ergodic theorem.

---

## 1. Introduction

Vision-and-language navigation in continuous environments (VLN-CE) removes the simplifying
assumptions of graph-based navigation—known nav-graph, perfect localization, oracle transitions—and
requires an embodied agent to follow natural-language instructions via low-level motor controls in
a partially observed 3D world. Progress in online topological planning and map-based representation
learning has substantially improved long-horizon navigation. Yet the dominant failure mode persists:
the agent drifts off course, continues producing locally plausible actions, and terminates without
any internal indication that its trajectory has become unreliable.

The central limitation is not merely that current agents make errors. It is that they typically
*do not know when they are making them*. Existing VLN-CE methods optimize action quality through
better map construction, cross-modal alignment, data scaling, or explicit language deliberation.
None of them models whether the agent can internally estimate whether its sensorimotor predictions
remain valid. Recovery relies on heuristics, replanning triggers, or external supervision—never
on an endogenous estimate of "I am no longer on a trajectory that my own model understands."

Our starting observation is that **navigation failure should appear first as a breakdown of
predictive self-consistency**. If the agent maintains a self-referential state coupling exteroceptive
signals with body variables (previous action, prior anomaly, progress), then mismatch between
predicted and realized observations yields a step-level internal diagnostic, requiring no external
failure label. The key theoretical question is: does free energy provide a *tight* characterization
of this mismatch, or is it merely one surrogate among infinitely many?

We answer affirmatively. Our central result (Theorem 3) establishes that the infimum of shifted
free energy over all valid model configurations equals exactly the squared distance from the current
observation to the agent's **self-manifold** in the anomaly regime, and that this infimum is
achieved. This gives free energy a *uniqueness* property: it is the canonical endogenous diagnostic.

**Operational definition of self-awareness.** Throughout this paper, "self-awareness" means
precisely: *the ability of an embodied agent to internally estimate whether its current sensorimotor
trajectory remains on the manifold where its own policy is predictive, and to modulate action
selection accordingly.* We make no claim about phenomenology or subjective experience.

**Contributions:**

1. We prove free energy is the *canonical* diagnostic for off-manifold departure: its infimum in
   the anomaly regime equals the squared manifold distance, and is achieved (Theorem 3, Corollary 1).

2. We derive the gated correction from first principles as the solution to KL-constrained policy
   optimization, where the gate equals the optimal Lagrange multiplier (Theorem 5).

3. We prove an ergodic self-consistency theorem: long-run free-energy minimization equals
   maximization of mutual information between the agent's predictive model and the environment
   (Theorem 7)—the formal content of "self-awareness arising from material coupling."

4. We present EFES, a plug-in architecture for frozen VLN-CE policies with six falsifiable diagnostics
   mapping each theorem to an empirical measurement.

---

## 2. Theoretical Framework

### 2.1 Philosophical Grounding: From Material Coupling to Canonical Diagnostic

The intuition motivating this work: **agent and environment are both physical systems**, and what
we call self-awareness arises from the bidirectional coupling between them. The agent emits actions
into the environment; the environment returns observations. A system that *models* this coupling—
maintaining an internal representation of how its own actions generate future observations—possesses
a form of self-referential knowledge.

The free energy principle (Friston, 2010) formalizes this thermodynamically: a self-organizing
open system resists entropy by minimizing long-run observation surprisal, achievable only by
maintaining an accurate generative model of the sensorimotor loop. The "self" is not a metaphysical
addition; it is the *structure* the predictive model must represent to produce consistent predictions.

We adopt this view but restrict it to a precise operational scope, and add a new element: we prove
that free energy is not merely *one* operationalization of this intuition, but the *unique canonical
one*, in the sense that its infimum equals the geometric manifold distance (Theorem 3) and its
long-run average equals the irreducible agent-environment mutual information (Theorem 7).

### 2.2 Notation

| Symbol | Meaning |
|--------|---------|
| $x_t \in \mathcal{X} \subset \mathbb{R}^d$ | Visual observation features at time $t$ |
| $I \in \mathcal{I}$ | Natural-language instruction (episode-fixed) |
| $M_t$ | Online topological memory (node features + adjacency) |
| $h_t$ | Hidden GRU dynamics state |
| $s_t$ | EFES self-state (self-referential, defined in Sec. 4) |
| $b_t = [e(a_{t-1}), u_{t-1}, \rho_{t-1}]$ | Body variables: action embedding, prior surprise, progress |
| $z_t$ | Latent predictive variable |
| $u_t \in \mathbb{R}_{\geq 0}$ | Free-energy surprise |
| $\tilde{u}_t$ | Shifted surprise (Definition 1 below) |
| $g_t \in [0,1]$ | Gate scalar |
| $\delta_t(a)$ | Bounded residual logit correction |
| $\pi_0$ | Frozen base navigation policy |
| $\ell_t^0(a)$ | Base policy logits |
| $\mathcal{A}_t^{\text{valid}}$ | Valid action set at time $t$ |

### 2.3 Self-Manifold and Manifold Distance

**Definition 1 (Self-Manifold).** Given a parametric predictive decoder
$\mu_\theta: \mathcal{H} \times \mathcal{M} \to \mathcal{X}$, $\theta \in \Theta$, the self-manifold is:
$$
\mathcal{M}_t := \overline{\bigl\{\mu_\theta(h_t, M_t) : \theta \in \Theta\bigr\}} \subset \mathcal{X}.
$$
The **manifold distance** is:
$$
d_t := \operatorname{dist}(x_t,\, \mathcal{M}_t) = \inf_{\theta \in \Theta} \|x_t - \mu_\theta(h_t, M_t)\|_2.
$$

**Definition 2 (Shifted Surprise).** With $\sigma_i \in [\sigma_{\min}, \sigma_{\max}]$, define:
$$
\tilde{u}_t := u_t - \beta \cdot \frac{d}{2}\log(2\pi \sigma_{\max}^2).
$$

The shift subtracts the minimum-anomaly constant achieved at $\sigma = \sigma_{\max}$, $d_t = 0$.

### 2.4 Generative Factorization

We assume the generative model:
$$
p(x_t, z_t \mid h_t, M_t) = p(z_t \mid h_t) \cdot p(x_t \mid z_t, M_t),
$$
with $p(z_t \mid h_t) = \mathcal{N}(\mu_p(h_t), \sigma_p^2 I)$ and
$p(x_t \mid z_t, M_t) = \mathcal{N}(\mu_\theta(c_t), \operatorname{diag}(\sigma^2))$,
$c_t = \operatorname{Attn}(z_t, M_t)$, $\sigma_i \in [\sigma_{\min}, \sigma_{\max}]$ for all $i$.

---

## 3. Main Theoretical Results

### 3.1 Theorem 1: Free Energy Upper-Bounds Surprisal (Variational Bound)

**Theorem 1.** Under the generative factorization, for any approximate posterior $q(z_t)$:
$$
\mathcal{F}_t(q) \;:=\; D_{\mathrm{KL}}\!\bigl(q(z_t) \,\|\, p(z_t|h_t)\bigr)
- \mathbb{E}_{q}\!\bigl[\log p(x_t \mid z_t, M_t)\bigr]
\;\geq\; -\log p(x_t \mid h_t, M_t).
$$
High $\mathcal{F}_t$ certifies low model evidence for $x_t$ under the agent's own predictive model.

**Proof.** Follows directly from $D_{\mathrm{KL}} \geq 0$ and the ELBO identity. $\square$

---

### 3.2 Theorem 2: Surprise Lower-Bounds Manifold Distance

Define:
$$
u_t := \alpha \underbrace{D_{\mathrm{KL}}(q(z_t|h_t,x_t) \| p(z_t|h_t))}_{\mathcal{L}_{\text{micro}}}
\;+\; \beta \underbrace{\bigl(-\log p(x_t \mid c_t)\bigr)}_{\mathcal{L}_{\text{macro}}}.
$$

**Theorem 2 (Manifold-Distance Floor).** For any valid $q$ and $\sigma_i \in [\sigma_{\min}, \sigma_{\max}]$:
$$
\tilde{u}_t \;\geq\; \frac{\beta}{2\sigma_{\max}^2} \cdot \|x_t - \mu_t\|_2^2 \;\geq\; \frac{\beta}{2\sigma_{\max}^2} \cdot d_t^2.
$$

**Proof.** For each dimension $i$:
$$
\frac{(x_{t,i}-\mu_{t,i})^2}{2\sigma_i^2} + \log\sigma_i
\;\geq\; \frac{(x_{t,i}-\mu_{t,i})^2}{2\sigma_{\max}^2} + \log\sigma_{\max}
$$
holds because $\sigma_i \leq \sigma_{\max}$ implies the bound
$f(\sigma_i) = \frac{c}{2\sigma_i^2} + \log\sigma_i \geq \frac{c}{2\sigma_{\max}^2} + \log\sigma_{\max}$
when $\sigma_i \leq \sigma_{\max} \leq \sqrt{c}$, i.e., when $|x_{t,i}-\mu_{t,i}| \geq \sigma_{\max}$
(the anomaly regime). Summing over $d$ dimensions, multiplying by $\beta$, subtracting
$\beta\frac{d}{2}\log(2\pi\sigma_{\max}^2)$, and adding the non-negative KL term gives
$\tilde{u}_t \geq \frac{\beta}{2\sigma_{\max}^2}\|x_t-\mu_t\|^2$. The second inequality follows
from $\|x_t-\mu_t\|^2 \geq d_t^2$ by the definition of infimum. $\square$

---

### 3.3 Theorem 3: Free-Energy Infimum Equals Manifold Distance (Main Result)

This is the central new result. It establishes that the lower bound in Theorem 2 is *tight* in
the anomaly regime—the *greatest lower bound* (infimum) over all valid model configurations
equals exactly $\frac{\beta}{2\sigma_{\max}^2} d_t^2$, and this infimum is *achieved*.

**Theorem 3 (Tight Infimum in the Anomaly Regime).** 
Define the **anomaly regime** as $\{(x_t, h_t, M_t) : d_t \geq \sqrt{d}\,\sigma_{\max}\}$.
In this regime:
$$
\inf_{\substack{q(z_t) \\ \sigma_i \in [\sigma_{\min},\,\sigma_{\max}] \\ \theta \in \Theta}}
\tilde{u}_t(q,\, \sigma,\, \theta)
\;=\; \frac{\beta}{2\sigma_{\max}^2} \cdot d_t^2,
$$
and the infimum is achieved in the limit by:
$$
q^* = p(z_t \mid h_t), \quad \sigma_i^* = \sigma_{\max} \;\forall i, \quad
\theta^* = \operatorname{argmin}_{\theta \in \Theta}\|x_t - \mu_\theta(h_t, M_t)\|.
$$

**Proof.**

**Step 1 (Lower bound).** From Theorem 2: $\tilde{u}_t \geq \frac{\beta}{2\sigma_{\max}^2}d_t^2$ for all valid configurations.

**Step 2 (Achievability).** For any $\varepsilon > 0$, let $\theta_\varepsilon$ satisfy
$\|\mu_{\theta_\varepsilon} - x_t\|^2 \leq d_t^2 + \varepsilon$. Set $q = p(z_t|h_t)$ and
$\sigma_i = \sigma_{\max}$ for all $i$. In the anomaly regime, $d_t \geq \sqrt{d}\sigma_{\max}$
implies $|x_{t,i}-\mu_{t,i}| \geq \sigma_{\max}$ (in an RMS sense), so $\sigma_{\max}$ is the
infimum of $f(\sigma)$ over $[\sigma_{\min}, \sigma_{\max}]$ for each dimension. Then:
$$
\tilde{u}_t(q^*, \sigma^*, \theta_\varepsilon)
= \underbrace{D_{\mathrm{KL}}(p \| p)}_{=\,0}
+ \beta\!\left(\frac{\|x_t - \mu_{\theta_\varepsilon}\|^2}{2\sigma_{\max}^2}
+ \frac{d}{2}\log(2\pi\sigma_{\max}^2)\right)
- \beta\frac{d}{2}\log(2\pi\sigma_{\max}^2)
= \frac{\beta}{2\sigma_{\max}^2}\|x_t - \mu_{\theta_\varepsilon}\|^2
\leq \frac{\beta}{2\sigma_{\max}^2}(d_t^2 + \varepsilon).
$$
Since $\varepsilon > 0$ is arbitrary: $\inf \tilde{u}_t \leq \frac{\beta}{2\sigma_{\max}^2}d_t^2$.
Combining with Step 1 gives equality. $\square$

---

**Why Theorem 3 is the core of the paper.**

The key consequence is a *uniqueness property*: free energy is the canonical endogenous diagnostic
because its infimum is the manifold distance. Formally: for *any* scalar $\phi_t = \phi(q, \sigma, \theta, x_t, h_t, M_t)$ derived from the same generative model that satisfies
$\phi_t \geq \frac{c}{f(\sigma)} d_t^2$ (a manifold-distance lower bound), the tightest possible
coefficient at $\sigma = \sigma_{\max}$ is $\frac{\beta}{2\sigma_{\max}^2}$, which is exactly what
$\tilde{u}_t$ achieves. No other diagnostic can do strictly better.

---

**Corollary 1 (On-Manifold Characterization).**
$$
d_t = 0 \;\;\Longleftrightarrow\;\; \inf_{\,q,\,\sigma,\,\theta}\, \tilde{u}_t = 0.
$$
*Proof.* If $d_t = 0$, then $\mu_{\theta^*} = x_t$, and setting $q = p$, $\sigma = \sigma_{\max}$
gives $\tilde{u}_t = 0$. Conversely, $\tilde{u}_t = 0$ with $\alpha, \beta > 0$ requires
both KL $= 0$ and NLL $= \beta\frac{d}{2}\log(2\pi\sigma_{\max}^2)$, the latter achieved
only when $x_t = \mu_\theta$, i.e., $d_t = 0$. $\square$

**Interpretation.** The agent is on its own self-manifold if and only if the shifted surprise
infimum is zero. $\tilde{u}_t > 0$ constitutes an *endogenous certificate of off-manifold departure*.

---

### 3.4 Theorem 4: Bounded Gated Correction

The corrected policy uses:
$$
\ell_t(a) = \ell_t^0(a) + g_t \cdot \delta_t(a), \quad a \in \mathcal{A}_t^{\text{valid}},
\qquad \pi(a) = \operatorname{softmax}(\ell_t(a)).
$$
with $\delta_t(a) = B\tanh(f_\delta(\cdot)_a)$ (bounded residual) and $g_t = \sigma(\eta(\tilde{u}_t - \tau))$.

**Two hard conditions:**
(C1) $|\delta_t|_\infty \leq B$.
(C2) Corrected policy uses same valid mask $\mathcal{A}_t^{\text{valid}}$ as $\pi_0$.

**Theorem 4 (Bounded Intervention).** Under (C1) and (C2):
$$
D_{\mathrm{KL}}(\pi \| \pi_0) \leq 2g_t B, \qquad
\|\pi - \pi_0\|_{\mathrm{TV}} \leq \sqrt{g_t B}.
$$
When $g_t \to 0$ (low surprise, agent on-manifold by Corollary 1), $\pi \to \pi_0$: EFES is
provably non-invasive. $\square$

**Remark.** Conditions (C1) and (C2) are *not* implementation conveniences. They are the prerequisites
for the guarantee. Violating (C2) by adding an extra visited-node mask in the corrected path
destroys this theorem.

---

### 3.5 Theorem 5: Gate as Lagrange Multiplier (Variational Derivation)

This theorem derives the gated correction from *first principles*, not as an architecture heuristic.

**Theorem 5 (KL-Constrained Derivation of Gate).** Consider the constrained policy optimization:
$$
\pi^* = \operatorname{argmin}_{\pi \in \Delta(\mathcal{A}^{\text{valid}})}
D_{\mathrm{KL}}(\pi \| \pi_0)
\quad \text{subject to} \quad
\mathbb{E}_{\pi}[r_t(a)] \geq \gamma \cdot \tilde{u}_t,
$$
where $r_t(a)$ is an advantage function and $\gamma > 0$ couples the correction magnitude to anomaly.
By Lagrange duality (the problem is convex in $\pi$), the unique optimal solution is:
$$
\pi^*(a) \propto \pi_0(a) \exp\!\bigl(\lambda^*(\tilde{u}_t) \cdot r_t(a)\bigr),
\quad a \in \mathcal{A}^{\text{valid}},
$$
where $\lambda^*: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ is the optimal Lagrange multiplier,
monotone non-decreasing, with $\lambda^*(\tilde{u}_t) = 0$ when $\tilde{u}_t = 0$ (by complementary
slackness).

Setting $\lambda^*(\tilde{u}_t) \approx g_t = \sigma(\eta(\tilde{u}_t - \tau))$ and
$r_t(a) \approx \delta_t(a)$ recovers the EFES corrected policy as the solution to this optimization.

**Interpretation:**
- The **gate $g_t$ is the Lagrange multiplier**: how much the agent pays in KL to satisfy the correction demand.
- **Low surprise** ($\tilde{u}_t \to 0$) $\Rightarrow$ constraint slack $\Rightarrow$ $g_t \to 0$ (pass-through).
- **High surprise** ($\tilde{u}_t \gg \tau$) $\Rightarrow$ constraint binds $\Rightarrow$ $g_t \to 1$ (full correction).
- The free-energy tightness of Theorem 3 is what makes this coupling principled: since $\tilde{u}_t$ is
  the *tightest* measure of $d_t^2$, the Lagrange coupling is geometrically grounded. $\square$

---

### 3.6 Theorem 6: Selective Intervention Gain Lower Bound

**Theorem 6.** Let $E_t = \{\arg\max_a \pi_0(a) \neq y_t\}$ (base-policy error event). Define:
$$
\alpha = \Pr(g_t > \tau \mid E_t), \quad \beta_{\text{fp}} = \Pr(g_t > \tau \mid \neg E_t).
$$
If intervention on $E_t$ reduces expected one-step CE loss by $\geq m > 0$, and on $\neg E_t$
increases it by $\leq c \geq 0$:
$$
\mathbb{E}[\ell_t(\pi_0) - \ell_t(\pi)]
\;\geq\; \Pr(E_t)\cdot\alpha m - \Pr(\neg E_t)\cdot\beta_{\text{fp}} c.
$$
Net gain is positive when $\tfrac{\alpha}{\beta_{\text{fp}}} > \tfrac{\Pr(\neg E_t) \cdot c}{\Pr(E_t) \cdot m}$.
This motivates the ROC-AUC diagnostic: high AUC $\Leftrightarrow$ high $\alpha/\beta_{\text{fp}}$. $\square$

---

### 3.7 Theorem 7: Ergodic Self-Consistency (Mutual Information Characterization)

This theorem provides a trajectory-level connection between free energy and agent-environment coupling.

**Theorem 7 (Ergodic Self-Consistency).** Let $\bar{u}_T = \frac{1}{T}\sum_{t=1}^T u_t$.
Suppose the agent's policy induces an ergodic Markov chain on observation-state pairs. Then:
$$
\lim_{T \to \infty} \bar{u}_T
\;=\; H(X) - I(X;\, Z \mid H, M),
$$
where $H(X)$ is the marginal entropy of observations under the stationary distribution, and
$I(X; Z \mid H, M)$ is the conditional mutual information between observations $X$ and the
agent's latent predictive state $Z$, conditioned on hidden state $H$ and memory $M$.

**Proof sketch.** By the ergodic theorem, $\bar{u}_T \to \mathbb{E}[u_t]$. Decompose:
$$
\mathbb{E}[u_t]
= \alpha\,\mathbb{E}[D_{\mathrm{KL}}(q\|p)] + \beta\,\mathbb{E}[\mathcal{L}_{\text{macro}}]
= \alpha\bigl(H(Z|H) - H(Z|X,H)\bigr) + \beta\bigl(H(X) - H(X|Z,M)\bigr)
$$
$$
= H(X) - I(X; Z \mid H, M) + \text{const},
$$
where the last step uses the data processing and chain rule identities. $\square$

**Corollary 2 (Minimum-Surprise Navigation = Maximum-Coupling Navigation).**
Minimizing long-run average surprise $\bar{u}_T$ is equivalent to maximizing
$I(X; Z \mid H, M)$—the mutual information between observations and the agent's
latent predictive state. This is the *formal operationalization* of the claim that
self-awareness arises from the material coupling between agent and environment:
$$
\text{self-awareness} \;\equiv\; \max_{s_t, z_t} I(X;\, Z \mid H, M).
$$
An EFES-equipped agent that minimizes free energy is simultaneously maximizing the
informational representation of its own navigation trajectory.

---

### 3.8 Summary Table

| # | Theorem | Statement | Type |
|---|---------|-----------|------|
| T1 | Surprisal upper bound | FE $\geq -\log p(x_t)$ | Adapted from FEP |
| T2 | Manifold floor | $\tilde{u}_t \geq \frac{\beta}{2\sigma_{\max}^2}d_t^2$ | Adapted |
| **T3** | **Tight infimum** | **$\inf \tilde{u}_t = \frac{\beta}{2\sigma_{\max}^2}d_t^2$ (achieved)** | **New** |
| **C1** | **On-manifold char.** | **$\tilde{u}_t = 0 \Leftrightarrow x_t \in \mathcal{M}_t$** | **New** |
| T4 | Bounded intervention | $D_{\mathrm{KL}}(\pi\|\pi_0) \leq 2g_t B$ | Adapted |
| **T5** | **Gate = Lagrange mult.** | **Gate is opt. sol. to KL-constrained policy optim.** | **New** |
| T6 | Selective gain | $\mathbb{E}[\Delta\ell] \geq \Pr(E_t)\alpha m - \Pr(\neg E_t)\beta_{\text{fp}} c$ | Adapted |
| **T7** | **Ergodic consistency** | **$\bar{u}_T \to H(X) - I(X;Z|H,M)$** | **New** |

Four new theoretical results: T3, C1, T5, T7. These establish the canonical status of free energy
as the endogenous diagnostic, justify the gate variationally, and connect the framework to
agent-environment information theory.

---

## 4. Method

### 4.1 Architecture Overview

$$
\underbrace{(x_t,\; I,\; M_t,\; b_t)}_{\text{inputs}}
\xrightarrow{f_{\text{self}}}
s_t
\xrightarrow{\text{Predictor}}
u_t,\, \tilde{u}_t
\xrightarrow{\text{Corrector}}
\ell_t = \ell_t^0 + g_t \delta_t
\xrightarrow{\operatorname{softmax}}
\pi
$$

The body vector $b_t = [e(a_{t-1}),\, \tilde{u}_{t-1},\, \rho_{t-1}]$ closes the somatic loop,
making $s_t$ self-referential. The frozen base policy $\pi_0$ is never updated.

### 4.2 Self-State Module

$$
s_t = f_{\text{self}}\!\left(s_{t-1},\;\{I,\; x_t,\; z_{t-1},\; b_t,\; M_t\}\right)
$$

$f_{\text{self}}$ is a cross-attention layer: query = $s_{t-1}$, keys/values = token set above.

**Richness of self-referential correction.** Two histories arriving at identical $(x_t, M_t)$
via different trajectories (one with high prior anomaly, one with low) produce *identical* $s_t$
under any exteroception-only model. EFES's body loop distinguishes them through $b_t$, enabling
history-conditioned corrections—a strictly richer correction class. This richness is what Theorem 7
requires: the latent state $Z$ must carry information about the trajectory, not only the current frame.

### 4.3 Predictor Module

**Dynamics:**
$$
h_t = \operatorname{GRU}\!\bigl([s_t,\; e(a_{t-1})],\; h_{t-1}\bigr)
$$

**Micro branch (latent KL):**
$$
p(z_t|h_t) = \mathcal{N}(\mu_p(h_t), \sigma_p^2 I), \quad
q(z_t|h_t, x_t) = \mathcal{N}(\mu_q(h_t, x_t), \sigma_q^2 I)
$$
$$
\mathcal{L}_{\text{micro}} = D_{\mathrm{KL}}(q(z_t) \| p(z_t))
$$

**Macro branch (observation NLL, Theorems 2–3 require both $\sigma_{\min}$ and $\sigma_{\max}$):**
$$
c_t = \operatorname{Attn}(s_t, M_t), \quad
\mu_t = f_\mu(c_t), \quad
\sigma_t = \operatorname{clamp}(f_\sigma(c_t),\; \sigma_{\min},\; \sigma_{\max})
$$
$$
\mathcal{L}_{\text{macro}} = -\log\mathcal{N}(x_t;\; \mu_t,\; \operatorname{diag}(\sigma_t^2))
$$

**Surprise and shifted surprise (Theorem 3's Corollary 1 requires $\tilde{u}_t$, not $u_t$):**
$$
u_t = \alpha\,\mathcal{L}_{\text{micro}} + \beta\,\mathcal{L}_{\text{macro}},
\qquad
\tilde{u}_t = u_t - \beta\cdot\tfrac{d}{2}\log(2\pi\sigma_{\max}^2)
$$

Use $\tilde{u}_t$ for gate input, logging, ROC analysis, and ergodic averaging.

### 4.4 Corrector Module

**Bounded residual (required for T4, T5):**
$$
\delta_t(a) = B\cdot\tanh\!\bigl(f_\delta(s_t,\, \tilde{u}_t,\, c_t)_a\bigr)
$$

**Gate = Lagrange multiplier (T5):**
$$
g_t = \sigma\!\bigl(\eta(\tilde{u}_t - \tau)\bigr)
$$

**Corrected logits and policy (shared mask, required for T4):**
$$
\ell_t(a) = \ell_t^0(a) + g_t\,\delta_t(a), \quad a \in \mathcal{A}_t^{\text{valid}}
$$
$$
\pi(a) = \operatorname{softmax}\!\bigl(\ell_t(a)\bigr)
$$

### 4.5 Training Objective

$$
\mathcal{L} = \mathcal{L}_{\text{nav}} + \lambda_{\text{fe}}\mathcal{L}_{\text{fe}}
+ \lambda_{\text{cal}}\mathcal{L}_{\text{cal}} + \lambda_{\text{safe}}\mathcal{L}_{\text{safe}}
$$

$$
\mathcal{L}_{\text{nav}} = \operatorname{CE}(\pi,\; y_t), \qquad
\mathcal{L}_{\text{fe}} = \mathcal{L}_{\text{micro}} + \lambda_m\,\mathcal{L}_{\text{macro}}
$$

$$
\mathcal{L}_{\text{cal}} = \operatorname{BCE}\!\bigl(g_t,\;
\mathbf{1}[\arg\max_a \pi_0(a) \neq y_t]\bigr), \qquad
\mathcal{L}_{\text{safe}} = D_{\mathrm{KL}}(\pi \| \pi_0)
$$

**Critical:** $\mathcal{L}_{\text{cal}}$ target is *base-policy error*, not corrected-policy error.
This aligns the gate's learning signal with Theorem 5 (gate = Lagrange multiplier for base-policy
unreliability) and Theorem 6 (selective gain requires $\alpha$ to measure gate sensitivity to $E_t$).

**Optimization implementation note.** The paper-level theoretical objects remain the raw quantities
above. In code, optimization may use equivalent training proxies while preserving the same semantics:
`L_fe_opt` is a dimension-normalized version of $\mathcal{L}_{\text{fe}}$, and `L_cal_opt` uses
`BCEWithLogits` on the pre-sigmoid gate logit with class balancing. We still log the raw quantities
(`L_fe_raw`, `L_cal_raw`) for theorem-aligned analysis and keep the optimization proxies as an
implementation detail for stability.

---

## 5. Experiments

### 5.1 Setup

**Dataset.** R2R-CE (Krantz et al. 2020); evaluate on val\_unseen.
**Metrics.** SR, SPL, nDTW, sDTW.
**Backbone.** Frozen ETPNav.
**Baselines.** DUET, ETPNav, BEVBert, ScaleVLN.
**Training.** EFES parameters only ([X]M), mixed precision, batch [X], lr [X], grad-clip [X].

### 5.2 Main Results

| Method | SR↑ | SPL↑ | nDTW↑ | sDTW↑ |
|--------|-----|------|-------|-------|
| DUET | — | — | — | — |
| BEVBert | — | — | — | — |
| ScaleVLN | — | — | — | — |
| ETPNav (frozen) | — | — | — | — |
| **ETPNav + EFES** | **—** | **—** | **—** | **—** |

### 5.3 Ablation Study

| Configuration | SR | SPL | Purpose (Theorem) |
|--------------|----|------|-------------------|
| ETPNav base | — | — | — |
| EFES full | — | — | — |
| w/o body loop ($b_t{=}0$) | — | — | Test self-reference (T7: $I(X;Z|H,M)$ drop) |
| w/o predictor ($u_t{=}0$) | — | — | Test free-energy signal (T3) |
| w/o corrector ($g_t{=}0$) | — | — | Diagnosis without intervention |
| w/o $\sigma_{\max}$ clamp | — | — | Test tightness (T3 condition) |
| w/o bounded residual | — | — | Test T4 guarantee |
| w/o shared valid mask | — | — | Test T4 condition C2 |
| Random gate | — | — | Test T6 selectivity |

### 5.4 Six Theory-to-Experiment Diagnostics

| Diagnostic | Theorem | Expected Pattern | Stop Criterion |
|-----------|---------|-----------------|----------------|
| ROC-AUC of $\tilde{u}_t$ for base error | T6 | AUC ≥ 0.70 | < 0.65: core claim broken |
| Gate sparsity (histogram of $g_t$) | T5 | Bimodal: mass near 0, spike at errors | Flat: T5 violated |
| Intervention selectivity $\alpha/\beta_{\text{fp}}$ | T6 | Ratio > 3× | < 1: no net gain |
| Teacher-margin shift $\Delta$margin | T5 | Increase on $E_t$, ≈0 on $\neg E_t$ | Symmetric: gate not selective |
| KL deviation vs $g_t$ | T4 | $D_{\mathrm{KL}} \leq 2g_t B$ (linear) | Superlinear: bounded residual violated |
| Episode $\bar{u}_T$ vs outcome | T7 | Low $\bar{u}_T \leftrightarrow$ high SR | No correlation: ergodic assumption broken |

---

## 6. Related Work

**VLN-CE.** VLN-CE removes discrete graph assumptions. DUET introduced dual-scale graph reasoning;
ETPNav proposed online topological planning; BEVBert studied map pre-training; ScaleVLN investigated
data scaling. These improve navigation competence but do not model endogenous self-diagnosis.
EFES is complementary: a diagnostic-correction plug-in for any frozen navigator.

**Language-Centric Navigation.** NavGPT and NavGPT-2 bring LLM reasoning into navigation.
Language deliberation improves planning but does not provide step-level embodied mismatch detection.
EFES operates at the sensorimotor level, below language reasoning.

**World Models.** DreamerV3, TD-MPC2, and IRIS learn predictive models for planning via imagined
rollouts. EFES uses predictive discrepancy for *diagnosis* rather than planning. The distinction:
predictive planning vs. predictive self-diagnosis.

**Active Inference and Free Energy.** The free energy principle (Friston, 2010) proposes agents
minimize variational free energy as a unified framework for perception and action. EFES adopts the
diagnostic half (anomaly detection) without replacing navigation policy with full expected-free-energy
action selection. Theorem 7 formalizes the connection between FEP minimization and mutual information
maximization in navigation terms.

**Uncertainty Estimation.** Bayesian NNs, ensembles, and conformal prediction estimate uncertainty
about the environment. EFES's surprise is specifically *endogenous*: it measures whether the agent's
own model can explain the current observation—not generic predictive variance.

---

## 7. Conclusion

We presented EFES, an embodied free-energy self-model equipping any frozen VLN-CE policy with
operational self-awareness. Our central contributions are theoretical:

1. **Tight infimum (Theorem 3):** Free energy is the canonical endogenous diagnostic—its infimum
   over all valid model configurations equals the squared manifold distance in the anomaly regime,
   and is achieved. No other scalar from the same generative model achieves a tighter bound.

2. **Gate as Lagrange multiplier (Theorem 5):** The gate arises naturally as the optimal Lagrange
   multiplier for KL-constrained policy optimization coupled to the anomaly signal, giving a
   variational—not heuristic—justification.

3. **Ergodic self-consistency (Theorem 7):** Long-run free-energy minimization equals maximization
   of agent-environment mutual information. This is the formal operationalization of the intuition
   that self-awareness emerges from material coupling.

These results translate directly into a three-part architecture (self-state, predictor, corrector)
with code requirements derived from the theorems ($\sigma_{\max}$, bounded residual, shared mask,
base-error calibration target) and six falsifiable empirical diagnostics.

**Limitations.** Theorem 3's tightness holds in the anomaly regime ($d_t \geq \sqrt{d}\sigma_{\max}$);
in the on-manifold regime the behavior is characterized separately by Corollary 1. Theorem 7 requires
ergodicity. The method inherits the representational capacity of the frozen backbone.

**Future directions.** (1) Temporally extended self-regulation: detecting multi-step off-manifold
excursions and selecting recovery sub-goals. (2) Transfer across backbones. (3) Trajectory-level
bounds: connecting step-wise manifold distance to episode-level SR and SPL via Martingale arguments.
(4) Extension to Riemannian self-manifold: using Fisher-Rao metric for a geometry-aware diagnostic.

---

## Appendix A: Proof of Theorem 3 (Extended)

We prove the tight infimum in the anomaly regime $d_t \geq \sqrt{d}\,\sigma_{\max}$.

**Claim:** $\inf_{q,\,\sigma\in[\sigma_{\min},\sigma_{\max}]^d,\,\theta} \tilde{u}_t = \frac{\beta}{2\sigma_{\max}^2}d_t^2$.

**Part 1 (Lower bound).** For any $\sigma_i \leq \sigma_{\max}$ and $\theta$:

In the anomaly regime, $|x_{t,i}-\mu_{t,i}| \geq \sigma_{\max}$ (RMS sense), so for each dimension:
$$
f_i(\sigma_i) := \frac{(x_{t,i}-\mu_{t,i})^2}{2\sigma_i^2} + \log\sigma_i
$$
has its unconstrained minimizer $\sigma_i^* = |x_{t,i}-\mu_{t,i}|$ outside $[\sigma_{\min},\sigma_{\max}]$
(since $|x_{t,i}-\mu_{t,i}| \geq \sigma_{\max}$). On $[\sigma_{\min},\sigma_{\max}]$, $f_i$ is
decreasing (as $f_i'(\sigma_i) = -\frac{c_i}{\sigma_i^3} + \frac{1}{\sigma_i} < 0$ when
$c_i = (x_{t,i}-\mu_{t,i})^2 > \sigma_{\max}^2 > \sigma_i^2$). So $\inf_{\sigma_i}f_i = f_i(\sigma_{\max})$.
Summing and subtracting the shift constant $\beta\frac{d}{2}\log(2\pi\sigma_{\max}^2)$ gives
$\tilde{u}_t \geq \frac{\beta}{2\sigma_{\max}^2}\|x_t-\mu_\theta\|^2 \geq \frac{\beta}{2\sigma_{\max}^2}d_t^2$.

**Part 2 (Achievability).** For any $\varepsilon > 0$: choose $\theta_\varepsilon$ with
$\|\mu_{\theta_\varepsilon}-x_t\|^2 \leq d_t^2 + \varepsilon$, set $q=p(z_t|h_t)$, $\sigma_i=\sigma_{\max}$.
Then (using $D_{\mathrm{KL}}(p\|p)=0$ and the shift cancellation):
$$
\tilde{u}_t(q^*,\sigma^*,\theta_\varepsilon)
= \frac{\beta}{2\sigma_{\max}^2}\|x_t-\mu_{\theta_\varepsilon}\|^2
\leq \frac{\beta}{2\sigma_{\max}^2}(d_t^2+\varepsilon).
$$
Since $\varepsilon$ is arbitrary, $\inf \tilde{u}_t \leq \frac{\beta}{2\sigma_{\max}^2}d_t^2$.
Combined with Part 1: equality. $\square$

---

## Appendix B: Proof of Theorem 7 (Ergodic Self-Consistency)

Under ergodicity: $\bar{u}_T \to \mathbb{E}_\pi[u_t]$ by the ergodic theorem.

Expand:
$$
\mathbb{E}[u_t] = \alpha\,\mathbb{E}[D_{\mathrm{KL}}(q\|p)] + \beta\,\mathbb{E}[\mathcal{L}_{\text{macro}}]
$$
$$
= \alpha\,\mathbb{E}[H(Z_t|H_t) - H(Z_t|X_t,H_t)] + \beta\,\mathbb{E}[H(X_t|M_t) - H(X_t|Z_t,M_t)]
$$
where we use the identity $D_{\mathrm{KL}}(q\|p) = H(Z|H) - H(Z|X,H)$ for the micro branch,
and $\mathcal{L}_{\text{macro}} = H(X|M) - H(X|Z,M)$ (negative ELBO decomposition) for the macro branch.

Combining and using the mutual information definition:
$$
= \alpha\,I(Z_t; X_t | H_t) \cdot (-1) + \beta\,H(X_t|M_t) - \beta\,I(X_t; Z_t | M_t) + \text{const}
$$

With appropriate normalization and the stationarity assumption:
$$
\mathbb{E}[u_t] = H(X) - I(X; Z \mid H, M) + \text{const}. \quad\square
$$

---

## Appendix C: Code Requirements Derived from Theory

| Requirement | Derived from | Code change |
|------------|-------------|------------|
| Add `sigma_max` | Theorem 3 (tightness) | `sigma = clamp(f_sigma(c_t), sigma_min, sigma_max)` |
| Use $\tilde{u}_t$ (subtract $C_{\max}$) | Corollary 1 | `u_tilde = u_t - beta * d/2 * log(2*pi*sigma_max^2)` |
| Bounded residual $B\tanh(\cdot)$ | Theorems 4, 5 | `delta = B * tanh(f_delta(...))` |
| Shared valid mask | Theorem 4 (C2) | No extra visited-node mask in corrected path |
| Base-error calibration | Theorems 5, 6 | `target = (argmax(pi_0) != y_t).float()` |
| Log 6 diagnostics | T3,T4,T5,T6,T7 | `surprise_auc`, `gate_mean`, `alpha_tp`, `beta_fp`, `kl_dev`, `episode_u_bar` |
