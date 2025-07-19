import os
import sys

# If not locally installed (or at least in editable mode), append path the base dir and src dir for module imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from insurance_mathematics.agg_dist.fft_nb import Agg_NbFft

from scipy import stats
import numpy as np

import unittest


def percentage_error(observed: float, expected: float) -> float:
    return np.abs(observed - expected) / expected


class Test_FFT_NB(unittest.TestCase):
    frequency: dict
    severity: dict
    layer: list
    agg_lim: float

    @classmethod
    def setUpClass(cls) -> None:
        # Set up parameters for the test class
        cls.frequency = {"dist": stats.nbinom, "properties": [3.0, 0.85]}

        cls.severity = {"dist": stats.lognorm, "properties": [1.2, 0.0, 7.8]}

        cls.layer = [3.0, 10.0]
        cls.agg_lim = 8.5
        return super().setUpClass()

    def test_ground_up(self):
        """Testing consistency of ground up FFT calculations vs exact."""
        n, p = self.frequency["properties"]
        mdl = Agg_NbFft(n=n, p=p, severity_distribution=self.severity)
        mdl.compile_aggregate_distribution()

        fft_mean = mdl.mean()
        exact_mean = mdl.mean(True)

        self.assertAlmostEqual(percentage_error(fft_mean, exact_mean), 0.0, 2)

    def test_layer(self):
        """Testing consistency of calculations in layer between fft and exact."""
        n, p = self.frequency["properties"]
        mdl = Agg_NbFft(n=n, p=p, severity_distribution=self.severity)

        mdl.setup_layer(*self.layer)
        mdl.compile_aggregate_distribution()

        fft_mean = mdl.mean()
        exact_mean = mdl.mean(True)

        self.assertAlmostEqual(percentage_error(fft_mean, exact_mean), 0.0, 2)

    def test_variance(self):
        """Test consistency of calcs for variance between FFT and exact."""
        n, p = self.frequency["properties"]
        mdl = Agg_NbFft(n=n, p=p, severity_distribution=self.severity)

        mdl.setup_layer(*self.layer)
        mdl.compile_aggregate_distribution()

        fft_var = mdl.var()
        exact_var = mdl.var(True)

        self.assertAlmostEqual(percentage_error(fft_var, exact_var), 0.0, 2)


if __name__ == "__main__":
    unittest.main()
