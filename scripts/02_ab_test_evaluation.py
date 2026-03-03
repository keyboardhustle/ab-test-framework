"""
A/B Test Evaluation Script
==========================
Usage:
  python 02_ab_test_evaluation.py --control CONVERSIONS,SESSIONS --variant CONVERSIONS,SESSIONS

Example:
  python 02_ab_test_evaluation.py --control 132,4120 --variant 159,4087

Outputs:
  - Chi-squared test (p-value)
  - Relative uplift and 95% confidence interval
  - Bayesian probability variant beats control
  - Business recommendation
"""

import argparse
import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True, help='conversions,sessions (e.g. 132,4120)')
    parser.add_argument('--variant', required=True, help='conversions,sessions (e.g. 159,4087)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default 0.05)')
    parser.add_argument('--mde', type=float, default=0.10, help='Minimum detectable effect as relative % (default 0.10)')
    parser.add_argument('--daily-traffic', type=int, default=None, help='Daily sessions to estimate business impact')
    parser.add_argument('--avg-deal-value', type=float, default=None, help='Avg deal value NOK for business impact')
    return parser.parse_args()


def frequentist_test(c_conv, c_n, v_conv, v_n, alpha=0.05):
    """Chi-squared test for proportions."""
    c_nonconv = c_n - c_conv
    v_nonconv = v_n - v_conv
    contingency = np.array([[c_conv, c_nonconv], [v_conv, v_nonconv]])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
    return chi2, p_value


def confidence_interval(c_conv, c_n, v_conv, v_n, confidence=0.95):
    """Wilson score confidence interval for the uplift."""
    p_c = c_conv / c_n
    p_v = v_conv / v_n
    uplift = (p_v - p_c) / p_c

    # Standard error of difference in proportions
    se = np.sqrt(p_c * (1 - p_c) / c_n + p_v * (1 - p_v) / v_n)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    diff = p_v - p_c
    ci_lower = diff - z * se
    ci_upper = diff + z * se
    return uplift, ci_lower / p_c, ci_upper / p_c


def bayesian_probability(c_conv, c_n, v_conv, v_n, n_samples=100_000):
    """Estimate P(variant > control) using Beta distribution sampling."""
    # Prior: Beta(1,1) = uniform
    control_samples = beta_dist.rvs(1 + c_conv, 1 + (c_n - c_conv), size=n_samples)
    variant_samples = beta_dist.rvs(1 + v_conv, 1 + (v_n - v_conv), size=n_samples)
    prob = np.mean(variant_samples > control_samples)
    return prob


def main():
    args = parse_args()

    c_conv, c_n = map(int, args.control.split(','))
    v_conv, v_n = map(int, args.variant.split(','))

    c_cvr = c_conv / c_n
    v_cvr = v_conv / v_n

    print("\n" + "=" * 60)
    print("A/B TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nControl:  {c_conv:,} conversions / {c_n:,} sessions = {c_cvr:.2%} CVR")
    print(f"Variant:  {v_conv:,} conversions / {v_n:,} sessions = {v_cvr:.2%} CVR")

    # Frequentist
    chi2, p_value = frequentist_test(c_conv, c_n, v_conv, v_n, args.alpha)
    uplift, ci_lower, ci_upper = confidence_interval(c_conv, c_n, v_conv, v_n)
    sig = p_value < args.alpha

    print(f"\n--- Frequentist Test ---")
    print(f"Relative uplift:    {uplift:+.1%}")
    print(f"95% CI (relative):  [{ci_lower:+.1%}, {ci_upper:+.1%}]")
    print(f"Chi-squared:        {chi2:.3f}")
    print(f"p-value:            {p_value:.4f}")
    print(f"Significant at {args.alpha:.0%}: {'YES' if sig else 'NO'}")

    # Bayesian
    prob_variant_wins = bayesian_probability(c_conv, c_n, v_conv, v_n)
    print(f"\n--- Bayesian Test ---")
    print(f"P(variant > control): {prob_variant_wins:.1%}")

    # MDE check
    meets_mde = abs(uplift) >= args.mde
    print(f"\n--- Business Significance ---")
    print(f"MDE threshold:     {args.mde:.0%} relative")
    print(f"Observed uplift:   {uplift:+.1%}")
    print(f"Meets MDE:         {'YES' if meets_mde else 'NO'}")

    if args.daily_traffic and args.avg_deal_value:
        extra_conv = (v_cvr - c_cvr) * args.daily_traffic * 30
        extra_pipeline = extra_conv * args.avg_deal_value
        print(f"Est. extra conversions/month: {extra_conv:.0f}")
        print(f"Est. pipeline impact/month:   NOK {extra_pipeline:,.0f}")

    # Decision
    print(f"\n--- RECOMMENDATION ---")
    if sig and meets_mde and prob_variant_wins > 0.95:
        print("SHIP THE VARIANT. Statistical + business significance confirmed. Bayesian confidence >95%.")
    elif sig and meets_mde:
        print("LEAN TOWARD SHIPPING. Statistically significant and meets MDE. Bayesian confidence moderate.")
    elif sig and not meets_mde:
        print("DO NOT SHIP. Statistically significant but uplift below MDE threshold. Not worth maintaining a variant.")
    else:
        print("DO NOT SHIP. No statistically significant result. Extend test or revisit hypothesis.")
    print()


if __name__ == '__main__':
    main()
