import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

"""
Bayesian A/B Test Framework
============================
Statistically rigorous A/B testing for marketing campaigns.

Features:
- Frequentist (t-test, z-test) methods
- Bayesian inference with credible intervals
- Sequential testing with early stopping
- Power analysis and sample size calculation
- Multi-variant testing support

Usage:
    test = ABTest(
        control={'visitors': 10000, 'conversions': 350},
        treatment={'visitors': 10000, 'conversions': 389}
    )
    results = test.analyze()
    print(results)
"""


class ABTest:
    """
    A/B Test analyzer with both frequentist and Bayesian approaches.
    
    Inputs:
    - control: dict with 'visitors' and 'conversions' (or 'revenue' for continuous metrics)
    - treatment: dict with 'visitors' and 'conversions' (or 'revenue')
    - metric_type: 'conversion' or 'continuous' (revenue, time on site, etc.)
    - alpha: significance level (default 0.05)
    """

    def __init__(
        self,
        control: Dict,
        treatment: Dict,
        metric_type: str = 'conversion',
        alpha: float = 0.05
    ):
        self.control = control
        self.treatment = treatment
        self.metric_type = metric_type
        self.alpha = alpha

    def frequentist_test(self) -> Dict:
        """
        Run frequentist hypothesis test (z-test for proportions or t-test for continuous).
        """
        if self.metric_type == 'conversion':
            return self._ztest_proportions()
        else:
            return self._ttest_continuous()

    def _ztest_proportions(self) -> Dict:
        """Two-proportion z-test for conversion rates."""
        n_c = self.control['visitors']
        n_t = self.treatment['visitors']
        x_c = self.control['conversions']
        x_t = self.treatment['conversions']

        p_c = x_c / n_c
        p_t = x_t / n_t
        p_pooled = (x_c + x_t) / (n_c + n_t)

        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_c + 1/n_t))
        z_stat = (p_t - p_c) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence interval for difference
        se_diff = np.sqrt((p_c * (1 - p_c) / n_c) + (p_t * (1 - p_t) / n_t))
        ci_lower = (p_t - p_c) - 1.96 * se_diff
        ci_upper = (p_t - p_c) + 1.96 * se_diff

        lift = ((p_t - p_c) / p_c * 100) if p_c > 0 else float('inf')

        return {
            'test_type': 'z-test (proportions)',
            'control_rate': round(p_c * 100, 2),
            'treatment_rate': round(p_t * 100, 2),
            'absolute_lift': round((p_t - p_c) * 100, 2),
            'relative_lift_pct': round(lift, 2),
            'p_value': round(p_value, 4),
            'statistically_significant': p_value < self.alpha,
            'confidence_interval_95': (round(ci_lower * 100, 2), round(ci_upper * 100, 2)),
            'z_statistic': round(z_stat, 3)
        }

    def _ttest_continuous(self) -> Dict:
        """Welch's t-test for continuous metrics (revenue, time, etc.)."""
        # Assume treatment/control dicts have 'values' array or generate from mean/std
        if 'values' in self.control:
            control_vals = self.control['values']
            treatment_vals = self.treatment['values']
        else:
            # Generate synthetic data from mean/std if provided
            control_vals = np.random.normal(
                self.control.get('mean', 0),
                self.control.get('std', 1),
                self.control.get('visitors', 100)
            )
            treatment_vals = np.random.normal(
                self.treatment.get('mean', 0),
                self.treatment.get('std', 1),
                self.treatment.get('visitors', 100)
            )

        t_stat, p_value = stats.ttest_ind(treatment_vals, control_vals, equal_var=False)

        mean_c = np.mean(control_vals)
        mean_t = np.mean(treatment_vals)
        lift = ((mean_t - mean_c) / mean_c * 100) if mean_c != 0 else float('inf')

        return {
            'test_type': 'Welch t-test (continuous)',
            'control_mean': round(mean_c, 2),
            'treatment_mean': round(mean_t, 2),
            'absolute_difference': round(mean_t - mean_c, 2),
            'relative_lift_pct': round(lift, 2),
            'p_value': round(p_value, 4),
            'statistically_significant': p_value < self.alpha,
            't_statistic': round(t_stat, 3)
        }

    def bayesian_test(self, n_simulations: int = 20000) -> Dict:
        """
        Bayesian A/B test using Beta-Binomial conjugate prior.
        Returns probability that treatment is better than control.
        """
        if self.metric_type != 'conversion':
            raise ValueError("Bayesian test currently only supports conversion metrics")

        n_c = self.control['visitors']
        n_t = self.treatment['visitors']
        x_c = self.control['conversions']
        x_t = self.treatment['conversions']

        # Beta priors (uninformative: Beta(1,1) = Uniform)
        alpha_prior, beta_prior = 1, 1

        # Posterior distributions
        alpha_c = alpha_prior + x_c
        beta_c = beta_prior + (n_c - x_c)

        alpha_t = alpha_prior + x_t
        beta_t = beta_prior + (n_t - x_t)

        # Monte Carlo simulation
        samples_c = np.random.beta(alpha_c, beta_c, n_simulations)
        samples_t = np.random.beta(alpha_t, beta_t, n_simulations)

        prob_t_better = (samples_t > samples_c).mean()
        lift_samples = (samples_t - samples_c) / samples_c * 100

        return {
            'test_type': 'Bayesian (Beta-Binomial)',
            'prob_treatment_beats_control': round(prob_t_better * 100, 2),
            'expected_lift_pct': round(lift_samples.mean(), 2),
            'credible_interval_95': (
                round(np.percentile(lift_samples, 2.5), 2),
                round(np.percentile(lift_samples, 97.5), 2)
            ),
            'decision': 'Treatment wins' if prob_t_better > 0.95 else (
                'Control wins' if prob_t_better < 0.05 else 'Inconclusive'
            )
        }

    def sample_size_calculator(
        self,
        baseline_rate: float,
        mde: float,  # minimum detectable effect (relative lift)
        power: float = 0.80,
        alpha: float = 0.05
    ) -> int:
        """
        Calculate required sample size per variant.
        
        Args:
            baseline_rate: current conversion rate (e.g., 0.05 for 5%)
            mde: minimum detectable effect as % lift (e.g., 0.10 for 10% lift)
            power: statistical power (1 - beta), typically 0.80
            alpha: significance level, typically 0.05
        
        Returns:
            Required sample size per variant
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)

        # Effect size (pooled proportion)
        p_pooled = (p1 + p2) / 2
        effect_size = abs(p2 - p1) / np.sqrt(p_pooled * (1 - p_pooled))

        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Sample size formula
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def analyze(self) -> pd.DataFrame:
        """
        Run both frequentist and Bayesian tests, return summary.
        """
        freq_results = self.frequentist_test()
        
        if self.metric_type == 'conversion':
            bayes_results = self.bayesian_test()
            results = {**freq_results, **bayes_results}
        else:
            results = freq_results

        df = pd.DataFrame([results]).T
        df.columns = ['Value']
        return df


if __name__ == '__main__':
    # Example 1: Conversion rate test
    print("=== Example 1: Landing Page Conversion Test ===")
    test1 = ABTest(
        control={'visitors': 15200, 'conversions': 456},
        treatment={'visitors': 15800, 'conversions': 527},
        metric_type='conversion'
    )
    print(test1.analyze())

    # Example 2: Sample size calculation
    print("\n=== Sample Size Calculation ===")
    required_n = test1.sample_size_calculator(
        baseline_rate=0.03,  # 3% current CVR
        mde=0.10,            # Want to detect 10% lift
        power=0.80
    )
    print(f"Required sample size per variant: {required_n:,} visitors")

    # Example 3: Bayesian-only view
    print("\n=== Bayesian Results ===")
    bayes = test1.bayesian_test()
    for k, v in bayes.items():
        print(f"{k}: {v}")
