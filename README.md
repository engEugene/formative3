# Formative 3 — Probability Distributions, Bayesian Inference & Gradient Descent

**Team:** Faith Mutoni · Shakira Munganyinka · Naillah Ineza · Eugene Koeach

This repository contains our full submission for Formative 3, covering four parts:

| Part | Topic | Folder |
|---|---|---|
| 1 | Expectation-Maximization (EM) for a Gaussian Mixture Model | [`part1/`](part1/) |
| 2 | Bayesian sentiment analysis on IMDb reviews | [`part2/`](part2/) |
| 3 | Manual gradient descent (matrix calculus) | [`part3/`](part3/) |
| 4 | Gradient descent in code, using SciPy for the derivative | [`part4/`](part4/) |

---

## Part 1 — EM Algorithm (`part1/`)

We model **Mothers vs. Adult Sons** heights from the Galton Families dataset as an unlabeled mixture of two 1-D Gaussians, and recover the two groups using Expectation-Maximization implemented from scratch (no `sklearn`).

**Why we compare mothers vs. sons (not all children):** all-children heights are themselves bimodal (male + female), which breaks the "two clean Gaussians" assumption. Restricting to sons keeps each component unimodal, which is what a 2-component GMM assumes.

**Run it:**
```bash
cd part1
pip install -r requirements.txt
python main.py
```
This prints the tracking table for iterations 0–2, then starts a live demo loop — type any height (e.g. the one the coach gives you) and see the exact posterior probabilities.

**Should we just split at the global mean?** No — and this is the core answer for the presentation. A hard split at the mean:
- Misclassifies every point in the overlap region (a tall son vs. a short mother near the boundary get the same treatment as points nowhere near it).
- Throws away the shape of the distributions — it doesn't care whether the data clusters tightly or spreads widely around each mean.
- Gives no measure of confidence. EM instead produces a **soft assignment** — e.g. a 68" data point might be 72% likely "Child" and 28% likely "Mother" — and uses that uncertainty itself to fit better parameters, rather than forcing a binary guess.

**Files:**
- `utils.py` — loads and combines the two unlabeled populations.
- `gaussian.py` — Gaussian PDF and log-likelihood.
- `em.py` — the E-step / M-step loop, with tracking table output.
- `main.py` — trains the model, prints the table, runs the live posterior demo.
- `presentation.ipynb` — the walkthrough notebook for the live presentation.

---

## Part 2 — Bayesian Sentiment Analysis (`part2/`)

Using the IMDb 50K review dataset, we compute **P(Positive | keyword)** via Bayes' Theorem for 3 positive-leaning and 3 negative-leaning keywords, in pure Python (no ML libraries).

**Keywords:** `excellent`, `masterpiece`, `wonderful` (positive-leaning) · `terrible`, `awful`, `worst` (negative-leaning)

**Run it:**
```bash
cd part2
python app.py
```

**Sample results:**

| Keyword | Prior P(Positive) | Likelihood P(kw\|Pos) | Marginal P(kw) | Posterior P(Pos\|kw) |
|---|---|---|---|---|
| excellent | 0.50 | 0.1147 | 0.0710 | **0.8074** |
| masterpiece | 0.50 | 0.0351 | 0.0241 | **0.7274** |
| wonderful | 0.50 | 0.0903 | 0.0556 | **0.8122** |
| terrible | 0.50 | 0.0153 | 0.0540 | **0.1418** |
| awful | 0.50 | 0.0114 | 0.0577 | **0.0985** |
| worst | 0.50 | 0.0164 | 0.0887 | **0.0927** |

Positive-leaning keywords push the posterior well above the 0.50 prior; negative-leaning keywords push it well below — exactly the direction Bayes' Theorem predicts.

---

## Part 3 — Manual Gradient Descent (`part3/`)

Manual, matrix-form derivation of gradient descent for `ŷ = Xm + b`, with 4 fully worked iterations (one per group member), using the chain rule to derive `∂J/∂m = (2/n)Xᵀe` and `∂J/∂b = (2/n)e`.

**MSE across iterations:** 61.00 → 6.50 → 2.50 → 2.16 → 2.10 (>96% error reduction in 4 steps).


---

## Part 4 — Gradient Descent in Code (`part4/`)

Converts the Part 3 derivation into working code. `scipy.optimize.approx_fprime` computes the gradient numerically (no hand-typed derivative formula), and every update step is explicit — nothing hidden behind a `.fit()` call.

**Run it:**
```bash
cd part4
pip install scipy matplotlib numpy
python part_4.py
```

This produces two plots, saved alongside the script:
- `params_over_iterations.png` — m and b drift over iterations
- `error_over_iterations.png` — MSE falling from 61.0 → ~2.1

Every printed iteration matches the Part 3 manual numbers exactly.

---

## Technologies

- Python 3, NumPy, Pandas, SciPy, Matplotlib
- No `sklearn` or other ML libraries used for the from-scratch parts (EM, Bayes)

## Team Contributions

- **Part 1 (EM):** Shakira Munganyinka
- **Part 2 (Bayes):** Eugene Koeach
- **Part 3 (Manual GD):** One iteration per member but Faith Mutoni worked on the rest.
- **Part 4 (Code):** Naillah Ineza

