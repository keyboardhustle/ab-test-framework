# A/B Test Framework for Marketing

## Problem

Marketing teams frequently make decisions based on underpowered tests, wrong statistical methods, or gut feel disguised as data. Common failure modes: calling a winner too early, ignoring business significance in favour of p-value, running tests too short, and misinterpreting results for non-normal metrics like revenue or form submissions.

This framework solves that: **a repeatable process for designing, running, and evaluating A/B tests on marketing assets** (landing pages, ad copy, email subject lines, CTAs).

## What This Framework Covers

| Step | What | Tool |
|------|------|------|
| 1. Design | Power analysis: sample size, MDE, duration | Python (statsmodels) |
| 2. Pre-flight | Sanity checks before launch | Python |
| 3. Evaluation | Frequentist (t-test, chi-squared) + Bayesian | Python (scipy, pymc) |
| 4. Interpretation | Business significance vs statistical significance | Markdown template |
| 5. Decision | Go / No-go framework with risk scoring | Markdown template |

## Key Files

```
/scripts
  01_power_analysis.py         → Calculate required sample size and test duration
  02_ab_test_evaluation.py     → Run frequentist + Bayesian evaluation on results
  03_sanity_checks.py          → Pre-launch and mid-test validity checks
/templates
  experiment_spec.md           → Test design document template
  results_readout.md           → Results interpretation template
README.md
```

## Worked Example: Landing Page CTA Test

**Hypothesis:** Changing the primary CTA from "Request a Demo" to "See It In Action" will increase demo form submissions by 15% or more.

**Test design:**
- Primary metric: demo form conversion rate
- Secondary metrics: scroll depth, time on page
- Traffic split: 50/50
- MDE: 15% relative lift (conservative; below this is not worth shipping)
- Baseline CVR: 3.2%
- Required sample: 3,840 sessions per variant
- Expected duration at 500 sessions/day: 8 days
- Significance threshold: p < 0.05, one-tailed

**Results:**

| Variant | Sessions | Conversions | CVR | Uplift |
|---------|----------|-------------|-----|--------|
| Control ("Request a Demo") | 4,120 | 132 | 3.20% | — |
| Variant ("See It In Action") | 4,087 | 159 | 3.89% | +21.5% |

- Chi-squared test: p = 0.018 → statistically significant
- Bayesian probability that variant beats control: 97.3%
- Business impact at current traffic: +27 demos/month, +~NOK 400k pipeline/month

**Decision: Ship the variant.** Both statistical and business significance thresholds met.

## Key Principles

1. **Always calculate sample size before running.** Never end a test because it "looks like" a winner.
2. **Pre-register your hypothesis and MDE.** Changing them after peeking invalidates the result.
3. **Check for novelty effects.** Re-evaluate after 2 weeks if you see an early spike.
4. **Business significance > statistical significance.** A 2% lift might be statistically real but not worth maintaining a variant.
5. **Document everything in the experiment spec template.** Institutional memory is part of the value.

## How to Run

```bash
pip install scipy statsmodels pandas numpy

# Step 1: Power analysis
python scripts/01_power_analysis.py --baseline 0.032 --mde 0.15 --alpha 0.05 --power 0.80

# Step 2: Evaluate results
python scripts/02_ab_test_evaluation.py --control 132,4120 --variant 159,4087
```

## Stack

- Python 3.11
- scipy (chi-squared, t-test)
- statsmodels (power analysis)
- pandas, numpy
