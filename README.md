# Optimal market making under inventory risk

Simulation code for **market making** with stochastic mid-price dynamics, Poisson-style order arrivals, and several quoting rules: a **fixed half-spread baseline**, a simple **inventory-aware heuristic**, and the **Avellaneda–Stoikov (AS)** closed-form approximation.

## Conceptual setup (Step 8)

A market maker posts a **bid** and **ask** and earns the spread when both sides trade, but must hold **inventory** `q` between trades. The mid-price moves randomly, so inventory is risky: if you are long and the price falls, mark-to-market wealth falls.

**Goal (intuitively):** choose quotes that balance **expected profit from spreads** against **inventory risk**, often expressed as maximizing expected utility of **terminal wealth** with risk aversion **γ**.

Two levers matter:

1. **Reservation price** — an “indifference” level for the asset that **shifts with inventory**: if you are too long, you tilt prices so that **selling is more attractive** and **buying less attractive**, and the opposite if you are short.

2. **Optimal spread** — how far you place the bid and ask around that reservation level. A **wider** spread captures more per trade but **reduces fill probability** (modeled here with decay **k**); **volatility**, **risk aversion**, and **time to horizon** all push spreads wider when inventory risk is more pressing.

You do not need the full HJB derivation in code; the classic **reduced formulas** below are the standard teaching shortcut.

## Avellaneda–Stoikov reduced formulas (Step 9)

Let **S_t** be the mid, **q_t** inventory, **σ** volatility, **γ > 0** risk aversion, **k** arrival sensitivity (consistent with **λ(δ) ∝ exp(−k·distance)** in the simulator), and **τ = T − t** time remaining.

**Reservation price**

\[
r_t = S_t - q_t \,\gamma\,\sigma^2\,\tau
\]

Interpretation: long inventory (**q > 0**) pulls the reservation price **down** so you quote to offload risk; short inventory pulls it **up**.

**Intuition (what each parameter does, Step 8–9)**

- **γ (risk aversion):** larger γ makes you care more about inventory variance. In the reduced AS formulas, that typically shows up as **wider** optimal spreads and **stronger** reservation-price tilting per unit of inventory — i.e. you become **less aggressive** on size (fewer marginal trades) to control risk.
- **σ (volatility):** larger σ means the mid is harder to predict; the **γ σ² τ** terms widen the spread and (through the reservation price) increase the incentive to **flatten** exposure before the horizon.
- **k (arrival sensitivity):** in **λ(δ) ∝ e^(−k·δ)**, larger k means counterparties are more sensitive to quote distance; optimal distances adjust through the **(1/γ) log(1 + γ/k)** term — effectively balancing **fill probability vs margin per trade**.
- **Baseline half-spread c:** for the fixed-spread control, **c** is the knob for **aggressiveness** independent of inventory math — smaller **c** → closer quotes → more fills (all else equal).
- **α (inventory penalty, heuristic):** larger α moves quotes to **relieve** a build-up in **q** (buy less aggressively when long, etc.), analogous in spirit to the AS reservation shift but in a simpler linear form.

**Symmetric half-spread** (common simplified form combining inventory-unconditional spread with a **γ σ² τ** term)

\[
\delta_t = \frac{1}{\gamma}\log\!\left(1 + \frac{\gamma}{k}\right) + \frac{\gamma\,\sigma^2}{2}\,\tau
\]

**Quotes**

\[
\text{bid} = r_t - \delta_t,\qquad \text{ask} = r_t + \delta_t
\]

The first term rewards balancing **fill sensitivity k** against **γ**; the second grows with **σ²** and **τ**, widening the quotes when the remaining horizon still exposes you to adverse moves.

Implementation: `src/strategy.py` (`AvellanedaStoikovStrategy`), wired through `MarketSimulator`.

## Parameter sensitivity (Step 10)

`src/sensitivity.py` sweeps one input at a time (holding others at defaults), runs **Monte Carlo** (`n_paths` replications per point), and saves figures under `results/sensitivity/`:

| Sweep | What to read |
|--------|----------------|
| **γ** | Higher γ → **wider** mean half-spread in simulation; aligns with **less aggressive** quoting from the AS δ formula. |
| **σ** | Higher σ → wider spreads (γσ²τ term); captures **vol-driven** caution. |
| **k** | Shifts the **log(1 + γ/k)** part of δ and interacts with the simulator fill model. |
| **c** (baseline) | Clear control of **distance to mid** without AS math. |
| **α** (heuristic) | Larger α → stronger lean against inventory (**mean \|q\|** proxy). |
| **T** (horizon) | Longer horizon changes **τ = T − t** weighting in both **r** and **δ** (and a separate plot illustrates reservation tilt vs **T** for a fixed reference inventory). |

Run (from `src/`):

```bash
python experiments.py sweep       # default ~120 paths per sweep point
python experiments.py sweep 200     # more paths → smoother curves (slower)
```

## Running experiments

From `src/` (so imports resolve):

```bash
python experiments.py              # sanity checks + one-path demos
python experiments.py mc          # Monte Carlo: baseline vs inventory-aware
python experiments.py mc 500    # same with 500 paths
python experiments.py mc as     # Monte Carlo: baseline vs Avellaneda–Stoikov
python experiments.py mc as 500
python experiments.py sweep 150   # parameter sweeps → results/sensitivity/
```

Figures are written under `results/` (see `src/plotting.py` and `src/sensitivity.py`).

## Project layout

- `src/market.py` — `MarketState` (cash **X**, inventory **q**, quotes, histories).
- `src/simulator.py` — mid-price shocks, quote update, Bernoulli fills, metrics history.
- `src/strategy.py` — quoting rules (baseline, heuristic, AS).
- `src/metrics.py` — per-path and Monte Carlo summaries (terminal PnL, drawdown, inventory stats).
- `src/experiments.py` — drivers and comparisons.
- `src/sensitivity.py` — Step 10 parameter sweeps and sensitivity figures.
