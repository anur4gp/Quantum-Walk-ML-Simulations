# CLAUDE.md

Project instructions for Claude Code. Read this fully before touching anything.

---

## 1. Project context

This is an ongoing computational physics research project in the group of **Dr. Chih-Chun Chien**, Department of Physics, University of California, Merced.

The project studies **discrete-time quantum walks (QWs) with classical randomness**, and uses **supervised machine learning classifiers** to locate the delocalization → localization transition automatically, instead of by manual inspection.

It continues two published papers from the group:

- **Paper A** — C. Mastandrea & C.-C. Chien, *"Localization of quantum walks with classical randomness: Comparison between manual methods and supervised machine learning,"* **Phys. Rev. E 108, 035308 (2023)**. This is the direct methodological ancestor of the current work. `PhysRevE_108_035308.pdf` (also present in the project as `ML_paper.pdf` — same document).
- **Paper B** — C. Mastandrea & C.-C. Chien, *"Robustness and classical proxy of entanglement in variants of quantum walks,"* **Phys. Rev. E 110, 064124 (2024)**. Source for the QW variants (conventional / symmetric / split-step), entanglement entropy, and the "overlap" proxy.

**Treat these two papers as the specification.** When there is a conflict between something you would guess and something stated in Paper A or Paper B, the papers win. When you implement a method that appears in a paper, cite the section in a code comment (e.g. `# Paper A, Sec. III B 2`).

### Current phase (what I'm actually doing right now)

Easing back into the school year. The QW simulation side is already working — I have reproduced pure Hadamard walks, spin-resolved distributions, discrete-random-coin walks, continuous-random-coin walks, and random-translation walks (see `figures/` and Section 4 below).

**Right now I am building and validating the ML classifiers in isolation, *before* wiring them to the transition-state figures / scaling analysis.**

That means, concretely:
- Build clean, testable `SVM`, `MLP`, and `CNN` classifier modules with a shared interface.
- Validate them on synthetic / trivially separable data first, then on real QW distributions from the far-delocalized and far-localized regimes.
- Confirm high train/test accuracy on the two-class problem before anything touches the transition regime.
- Only **after** that is settled: sweep the intermediate regime, extract critical values, produce the transition figures and scaling exponents.

**Do not skip ahead.** If I ask for a classifier, build the classifier — do not also generate the full scaling pipeline and transition plots unless I explicitly ask. Premature integration is the main thing I'm trying to avoid this phase.

---

## 2. Physics background and conventions

Keep these definitions straight; a lot of subtle bugs come from mixing conventions between Paper A and Paper B.

### 2.1 The walk

A walker on a 1D lattice with two internal ("coin" / "spin") states $|+\rangle, |-\rangle$:

$$|\psi\rangle = \sum_x \left(\alpha_{x,+}|x,+\rangle + \alpha_{x,-}|x,-\rangle\right)$$

**Coin (rotation) operator:**

$$\hat{C}(\theta,\phi_1,\phi_2) = \begin{pmatrix} \cos\theta & e^{i\phi_1}\sin\theta \\ e^{i\phi_2}\sin\theta & -e^{i(\phi_1+\phi_2)}\cos\theta \end{pmatrix}$$

Throughout both papers $\phi_1 = \phi_2 = \pi/2$, which reduces this to

$$\hat{C}(\theta) = \begin{pmatrix} \cos\theta & i\sin\theta \\ i\sin\theta & \cos\theta \end{pmatrix}$$

**Default $\theta_0 = \pi/6$** unless stated otherwise. Paper A confirms $\Delta\theta_c$ is insensitive to $\theta_0$ as long as $\theta_0$ is away from $0$ and multiples of $\pi/2$.

**Conventional translation operator:** $\hat{T}|\psi_x,+\rangle = |\psi_{x+1},+\rangle$, $\hat{T}|\psi_x,-\rangle = |\psi_{x-1},-\rangle$.

**Evolution:** $|\psi(t)\rangle = (\hat{T}\hat{C})^N|\psi_0\rangle$.

### 2.2 Lattice conventions — READ THIS, IT BITES

The two papers use **different lattices**:

| | Paper A (PRE 108) | Paper B (PRE 110) |
|---|---|---|
| Lattice | odd, $(2N+1)$ sites, centered on $x=0$ | even, $2N$ sites, **no** $x=0$ |
| Initial state | $\frac{1}{\sqrt{2}}(\vert 0,+\rangle + \vert 0,-\rangle)$ | $\sum_{x=\pm1,\sigma=\pm}\frac{1}{2}\vert x,\sigma\rangle$ |
| Max steps | $N$ | $N-1$ (conventional/symmetric), $N/2 - 1$ (split-step) |

Paper B needs the even lattice because the **symmetric** and **split-step** translation operators are *not unitary* on a lattice with a central $x=0$ site.

**The current ML work follows Paper A's conventions** (odd lattice, centered start, conventional walk) because that is the localization-transition setting. If a module ever needs Paper B's variants, it must take the lattice parity as an explicit parameter — never hardcode.

Assert unitarity in tests: total probability conserved to $\sim10^{-12}$.

### 2.3 The three types of classical randomness (Paper A, Sec. II B)

These are the three "channels" of randomness. Every classifier experiment is run separately for each.

1. **Discrete random rotation** (`discrete_coin`, my figures call it "Jittered")
   Two coin operators with $\theta_{1,2} = \theta_0 \pm \Delta\theta$. At each time step a **fair classical coin** picks one: $P(\theta_1) = P(\theta_2) = 1/2$.
   Control parameter: $\Delta\theta$, scanned over $0 \le \Delta\theta \le \theta_0$.

2. **Continuous random rotation** (`continuous_coin`, my figures call it "Uniform Jittered")
   $\theta(t) = \theta_0 + \Delta\theta(t)$ with $\Delta\theta(t) \sim \mathrm{Uniform}(0, \Delta\theta_M)$, redrawn each step.
   Control parameter: $\Delta\theta_M$.

3. **Random translation** (`random_translation`, my figures call it "Random Transform")
   One fixed coin. An inverse translation $\hat{T}^{-1}$ ($|\psi_x,+\rangle \to |\psi_{x-1},+\rangle$, $|\psi_x,-\rangle \to |\psi_{x+1},-\rangle$) is applied with probability $P_r$ instead of $\hat{T}$, decided each step.
   Control parameter: $P_r \in [0, 0.5]$. Above $0.5$ it's mirror-symmetric to $1-P_r$ by parity.

All three are **temporal** (redrawn each time step), uniform in space. Paper B additionally has **spatially dependent** randomness (drawn once per site, frozen in time) — not part of the current ML phase, but the code should not make temporal randomness structurally impossible to swap out.

### 2.4 Phenomenology to expect

- **Weak randomness → delocalized:** the signature two-peak (ballistic) distribution, MoI $\propto N^2$.
- **Strong randomness → localized:** single central peak, Gaussian-like (time-dependent randomness) or exponential (spatially dependent).
- **At the transition:**
  - discrete & continuous rotation → distribution **flattens out**
  - random translation → **three-peak structure** (central peak coexisting with two side peaks)

That three-peak structure is exactly what broke the NN methods in Paper A. Expect it; don't "fix" it.

### 2.5 Manual (baseline) observables

These are the ground-truth comparators the ML must be checked against.

- **Probability distribution:** $P(x) = |\psi_+(x)|^2 + |\psi_-(x)|^2$. Transition located visually by flattening / three-peak onset.
- **Moment of inertia:** $\mathrm{MoI}(t) = \sum_x x^2 P(x,t)$. Delocalized $\Rightarrow$ MoI $\propto N^2$. Critical value = the **kink** where it departs from $N^2$.
- **Inverse participation ratio:**
  $$\mathrm{IPR} = \frac{\left(\sum_x |\langle\psi_x,+|\rangle|^2\right)^2}{\sum_x |\langle\psi_x,+|\rangle|^4}$$
  Critical value = the **maximum** of IPR vs. randomness (flattest distribution). Note IPR **cannot** distinguish one-peak from two-peak — it only flags flatness. Do not oversell it.

### 2.6 Entanglement quantities (Paper B — not in current phase, keep for later)

- **Entanglement entropy:** $ES(t) = -\mathrm{Tr}_s(\rho_s\ln\rho_s) = -\lambda_+\ln\lambda_+ - \lambda_-\ln\lambda_-$, with $\rho_s = \mathrm{Tr}_x\rho_{tot}$. **Natural log**, base $e$ — differs from base-2 results in the literature by $\approx 0.69$.
- **Overlap (classical proxy):** $O(t) = \sum_x P_+(x,t)P_-(x,t)$. Tracks the *inverse* of $ES$. Steady-state values are computed as long-time averages, after transients.

---

## 3. Published results to reproduce / beat

Paper A, Table I — power-law exponents of the critical randomness vs. system size, $\Delta\theta_c \sim N^{-\alpha}$:

| Method | $\Delta\theta_c$ (discrete) | $\Delta\theta_c$ (continuous) | $P_{r,c}$ |
|---|---|---|---|
| SVM | 0.36 ± 0.04 | 0.25 ± 0.02 | 0.28 ± 0.04 |
| MLP NN | 0.32 ± 0.02 | 0.25 ± 0.01 | 0.07 ± 0.06 |
| CNN | 0.39 ± 0.07 | 0.32 ± 0.04 | −0.08 ± 0.06 |
| MoI | 0.62 ± 0.02 | 0.36 ± 0.02 | 0.51 ± 0.04 |
| Human | 0.32 ± 0.01 | 0.36 ± 0.01 | 0.69 ± 0.03 |
| IPR | 0.46 ± 0.02 | 0.41 ± 0.01 | 0.52 ± 0.02 |

**Key known failure mode:** for random translation, MLP and CNN systematically deviate (CNN even goes negative). Paper A tried splitting the distribution into three spatial regions ($-N<x<-N/2$, $-N/2<x<N/2$, $N/2<x<N$) as separate training sets and found **no improvement**. Don't re-run that experiment expecting a different answer unless we're deliberately extending it.

**This failure is the scientific opening for the current work.** If a new architecture or representation fixes the random-translation exponent, that is the result. Treat it as the target, not as a bug to paper over.

---

## 4. Existing assets

In `figures/` (or the project root) there are already-generated reference plots. Match their conventions when producing new ones.

| File | What it shows |
|---|---|
| `Total_probabilities.png` | Hadamard QW, 300 steps, total $P(x)$ — clean two-peak |
| `Spin_probabilities.png` | $P_+$ (blue solid) vs $P_-$ (red dashed), 300 steps — peaks at $x\approx\pm210$, components avoid each other |
| `Pure_Coin_vs__Random_Coin.png` | Pure QW vs discrete-random ("Jittered") vs continuous-random ("Uniform Jittered"), 300 steps |
| `Randoms_vs__Control.png` | All of the above plus random translation ("Random Transform"), 300 steps |
| `Peak_spin_updown_theta_vs_distance.png` | Peak spin-up position $\Delta x_m$ vs coin angle $\theta \in [-\pi/2,\pi/2]$; max $\approx 300$ at $\theta=0$, $\to 0$ at $\theta=\pm\pi/2$ |

My internal shorthand, for consistency in code and plot labels:

| My label | Formal name | Code key |
|---|---|---|
| Pure QW | no classical randomness | `pure` |
| Jittered | discrete random rotation | `discrete_coin` |
| Uniform Jittered | continuous random rotation | `continuous_coin` |
| Random Transform | random translation | `random_translation` |

---

## 5. Repository layout

Expected structure. **Reconcile with what's actually on disk before assuming** — run `ls` / `tree` first. If reality differs, tell me rather than silently restructuring.

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── src/
│   ├── qw/
│   │   ├── operators.py      # coin, translation, inverse translation
│   │   ├── walk.py           # evolution driver; returns psi_+, psi_-
│   │   ├── randomness.py     # the three randomness channels
│   │   └── observables.py    # P(x), MoI, IPR, (later) ES, overlap
│   ├── data/
│   │   ├── generate.py       # sample generation, labeling, caching
│   │   └── preprocess.py     # normalization to Pmax = 1
│   ├── models/
│   │   ├── base.py           # shared Classifier interface
│   │   ├── svm.py
│   │   ├── mlp.py
│   │   └── cnn.py
│   ├── analysis/
│   │   ├── critical.py       # confusion-point extraction  [LATER PHASE]
│   │   └── scaling.py        # power-law fits              [LATER PHASE]
│   └── plotting/
│       └── style.py
├── scripts/                  # thin CLI entry points, no logic
├── tests/
├── data/                     # cached .npz, gitignored
└── figures/
```

Physics lives in `src/qw/`. ML lives in `src/models/`. **They talk only through arrays.** No model should ever import a walk simulator directly.

---

## 6. Data contract

Fix this now so nothing downstream has to guess.

**A sample** = the final-time probability distribution of one simulation run.

- Shape: `(2N+1,)` float64 for the Paper A odd lattice.
- One run = one realization of the classical randomness (Monte Carlo over the classical coin), evolved to fixed $N$.
- Metadata carried alongside every sample: `channel`, `theta_0`, `N`, `control_value` ($\Delta\theta$ / $\Delta\theta_M$ / $P_r$), `seed`.

**Labels:** `0 = delocalized` (weak randomness), `1 = localized` (strong randomness). Never flip this.

**Normalization (important, Paper A Sec. III B 4):**
- Physical normalization is $\sum_x P(x) = 1$.
- **ML normalization is $P_{max} = 1$** — divide each distribution by its own maximum, matching the image-recognition convention.
- Apply to **both** training and inference data.
- Paper A found this doesn't help the SVM but substantially reduces variance for MLP and CNN. Keep it on by default for all three; make it a flag, not a hardcode.

**Training-window selection:** the ranges of the control parameter used to generate the two labeled classes must be tuned. Too narrow → not enough information about the configurations. Too wide → transition-regime data leaks into training and corrupts the critical-value estimate. Record the chosen windows in the run config; they are a result, not an implementation detail.

**Sample size:** Paper A settled on **1800 training samples** (stable estimates from ~2000; more gave no visible improvement). Use 1800 as the default. Train/test split **80/20**.

**Caching:** simulations are the expensive part. Cache to `data/*.npz` keyed by a hash of the config. Never regenerate silently — log a cache hit or miss.

---

## 7. Classifier specifications

All three implement the same interface (`src/models/base.py`):

```python
class Classifier(Protocol):
    def fit(self, X, y) -> None: ...
    def predict_proba(self, X) -> np.ndarray:  # shape (n, 2)
        ...
    def score(self, X, y) -> float: ...
```

`predict_proba` returning genuine two-class probabilities is non-negotiable — the whole critical-point method depends on it.

### 7.1 SVM (Paper A, Sec. III B 1)

- `sklearn` linear **SGDClassifier**
- Loss: **`modified_huber`** — *not* hinge. Hinge gives hard true/false output; modified Huber yields calibrated binary probabilities.
- Baseline expectation: near-perfect test accuracy even at small sample size (the two classes are one-peak vs two-peak, trivially separable). High accuracy here proves nothing except that the pipeline isn't broken. Say so in any summary you write.

### 7.2 MLP (Paper A, Sec. III B 2)

- `sklearn` **MLPClassifier**
- Six layers total: input (auto-sized to lattice) → hidden **400, 200, 100, 50** → output **2** neurons
- Regularization $\alpha = 0.001$
- Hyperparameters were explored with `GridSearchCV` over hidden layer sizes and $\alpha$
- These layer sizes were kept fixed from $N=80$ to $N=1000$; scaling them with lattice size gave negligible improvement

### 7.3 CNN (Paper A, Sec. III B 3)

- **TensorFlow/Keras**
- Four **1D convolutional** layers; a final **dense** layer with 2 neurons and **softmax**
- Filter counts scale with lattice size: layer 1 = $2N+1$, then half, quarter, eighth (round up)
- **Kernel size 3** — chosen because the lattice size is odd, so no padding is needed
- **Dropout** after each layer: 0.5, 0.5, then 0.2 on the final layer
- Loss: **SparseCategoricalCrossentropy**
- **10 epochs**
- For **random translation only**: add **L2 kernel regularization, $\lambda = 0.01$**, to every layer. This was necessary to control overfitting on the more varied distributions, and helped most at smaller $N$ (80–500).
- Model must be **retrained per lattice size** (filter counts change). Network *structure* stays fixed.

### 7.4 Critical-point extraction — LATER PHASE, but define it now

After training on the two extreme regimes, feed intermediate-randomness data and read off the transition:

- **SVM:** the point of **maximal confusion** — where the two class probabilities are equal.
- **MLP and CNN:** these show a *sudden* jump rather than a gradual crossover. Use the **first point where $P(\text{delocalized})$ falls below 50%**.

Do not use one rule for all three. This asymmetry is deliberate and is stated in the paper.

Known artifact: the ML methods estimate from a **small range** around the transition rather than a single point, which is the suspected cause of their systematically lower exponents.

---

## 8. Code conventions

- **Python 3.11+**, NumPy-first. Complex amplitudes are `complex128`. Probabilities `float64`.
- Type hints on all public functions. Docstrings: what it computes, and the paper section it comes from.
- **No global state. No hidden singletons.** Every function that uses randomness takes an explicit `rng: np.random.Generator` or `seed: int`. Never call `np.random.*` at module level.
- **Reproducibility is mandatory.** Every figure and every number must be regenerable from a config + seed. Seed NumPy, sklearn (`random_state`), and TensorFlow.
- Prefer vectorized NumPy over Python loops for the lattice; a Python loop over *time steps* is fine and expected.
- Config as plain dicts or dataclasses serialized to YAML/JSON in `configs/`. No argparse sprawl.
- Scripts in `scripts/` are thin — parse args, load config, call into `src/`. No physics in scripts.

### Dependencies

Current stack: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow`, `pytest`.
**Ask before adding anything else.** Especially: don't reach for PyTorch, JAX, or a plotting library that isn't matplotlib without discussing it — reproducing Paper A means matching its stack.

### Testing

`pytest`, in `tests/`. Minimum coverage before the ML work is trusted:

- **Unitarity:** $\sum_x P(x) = 1$ to $10^{-12}$ for every walk variant at every step.
- **Pure Hadamard walk** at $N=300$, $\theta=\pi/4$: two symmetric peaks, ballistic spread.
- **Known limits:** conventional walk maximal spreading at $\theta = n\pi$, minimal at $\theta = n\pi/2$ (odd $n$). Peak position vs $\theta$ should reproduce `Peak_spin_updown_theta_vs_distance.png`.
- **$\hat{T}^{-1}\hat{T} = \mathbb{1}$.**
- **$P_r = 0$** random-translation walk is bitwise identical to the pure walk.
- **Determinism:** same seed → identical arrays.
- **Symmetry:** for the symmetric initial condition with $\phi_1=\phi_2=\pi/2$, $P_+ = P_-$ totals.

### Plotting

- Match the existing figures: matplotlib defaults, `x` on the horizontal, `P(x)` vertical, descriptive title with step count.
- Line styles by channel: pure = blue solid, discrete = red dashed, continuous = green dotted, random translation = purple dash-dot. This matches `Randoms_vs__Control.png` — keep it consistent so figures are comparable across the project.
- Save as PNG at publication DPI (≥300) into `figures/`. Never overwrite an existing figure without saying so.

---

## 9. Working agreement

**Scope discipline.** Do what I asked. If you notice something adjacent that should be fixed, say so at the end — don't just do it. This project has a specific sequencing (classifiers first, then transition figures) and unrequested integration work actively costs me.

**Physics fidelity over convenience.** Do not simplify a physical definition to make code cleaner. If MoI, IPR, the overlap, or an operator definition would be easier with a different convention, that's not a reason to change it. Ask.

**Don't fabricate.** If you don't know what a paper says, say so and I'll check, or extract it from the PDFs in the project. Don't invent hyperparameters, exponents, or citations. If you're inferring rather than reading, mark it clearly.

**No silent restructuring.** Don't rename files, reorganize directories, or reformat code I didn't ask you to touch.

**Numerical claims need provenance.** If you report an accuracy, a critical value, or an exponent, include the config and seed that produced it.

**Flag known-hard cases.** Random translation is expected to be difficult. If a random-translation classifier gives a suspiciously clean result, be suspicious — check for label leakage or a training window that's swallowed the transition regime.

**Ask when the answer is a research decision.** Choosing a training window, an architecture change, or a new representation is my call and Dr. Chien's, not a coding decision.

---

## 10. Roadmap

### Phase 1 — Classifiers in isolation ← **CURRENT**
- [ ] Shared `Classifier` interface
- [ ] SVM implementation matching Paper A spec
- [ ] MLP implementation matching Paper A spec
- [ ] CNN implementation matching Paper A spec
- [ ] Data generation + caching for the two extreme regimes
- [ ] $P_{max}=1$ normalization applied and toggleable
- [ ] Test suite green (unitarity, known limits, determinism)
- [ ] Train/test accuracy verified for all three, all three randomness channels
- [ ] Sample-size sensitivity check reproducing Paper A Fig. 4(a),(c)

### Phase 2 — Transition regime
- [ ] Intermediate-randomness sweeps
- [ ] Confusion-point extraction (SVM rule and NN rule, separately)
- [ ] Transition-state figures

### Phase 3 — Scaling
- [ ] Critical value vs $N$, log-log
- [ ] Power-law fits with uncertainties
- [ ] Comparison table against Paper A Table I

### Phase 4 — Extension (the actual new science)
- [ ] Investigate why NN methods fail on random translation
- [ ] Test representations / architectures beyond Paper A
- [ ] Possible bridge to Paper B quantities (ES, overlap) as classifier inputs or targets

---

## 11. Glossary

| Term | Meaning |
|---|---|
| QW | Quantum walk (discrete-time) |
| Coin / rotation operator | $\hat{C}(\theta)$, mixes internal states |
| Translation operator | $\hat{T}$, shifts $\vert+\rangle$ right, $\vert-\rangle$ left |
| $\Delta\theta$ | Discrete random rotation strength |
| $\Delta\theta_M$ | Continuous random rotation upper bound |
| $P_r$ | Probability of applying inverse translation |
| $N$ | Number of time steps (and sets lattice size) |
| MoI | Moment of inertia, $\sum_x x^2 P(x)$ |
| IPR | Inverse participation ratio |
| ES | Entanglement entropy (natural log) |
| Overlap $O$ | $\sum_x P_+ P_-$, classical proxy for entanglement |
| SPE | Single-particle entanglement (internal ↔ positional) |
| Confusion point | Where classifier outputs equal class probabilities → critical value |