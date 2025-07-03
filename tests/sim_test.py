import os
import sys

# If not locally installed (or at least in editable mode), append path the base dir and src dir for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from insurance_mathematics.agg_dist.agg_sim import AggSim
from scipy import stats
import numpy as np

import unittest


def in_tol(observed: float, expected: float) -> bool:
    error = np.abs(observed - expected) / expected
    return error < 0.1


class Test_Sim(unittest.TestCase):
    freq_poisson: dict
    severity: dict
    layer: list

    # NOTE: Turn off parallel proceessing here for tests - joblib parallel may not allow isolation
    # which unittest requires.

    @classmethod
    def setUpClass(cls) -> None:
        cls.freq_poisson = {
            'dist': stats.poisson,
            'properties': [0.5]
        }

        cls.severity = {
            'dist': stats.lognorm,
            'properties': [1.2, 0.0, 7.8]
        }

        cls.layer = [3.0, 10.0]

        return super().setUpClass()

    def test_ground_up(self):
        '''Testing consistency of ground up FFT calculations vs exact.'''
        mdl = AggSim(frequency_distribution=self.freq_poisson,
                     severity_distribution=self.severity,
                     parallel=False)
        mdl.compile_aggregate_distribution()

        sim_mean = mdl.mean()
        exact_mean = mdl.mean(True)

        self.assertTrue(in_tol(sim_mean, exact_mean))

    def test_layer(self):
        '''Testing consistency of calculations in layer between fft and exact.'''
        mdl = AggSim(frequency_distribution=self.freq_poisson,
                     severity_distribution=self.severity,
                     parallel=False)

        mdl.setup_layer(*self.layer)
        mdl.compile_aggregate_distribution()

        sim_mean = mdl.mean()
        exact_mean = mdl.mean(True)

        self.assertTrue(in_tol(sim_mean, exact_mean))

    def test_variance(self):
        '''Test consistency of calcs for variance between FFT and exact.'''
        mdl = AggSim(frequency_distribution=self.freq_poisson,
                     severity_distribution=self.severity,
                     parallel=False)

        mdl.setup_layer(*self.layer)
        mdl.compile_aggregate_distribution()

        sim_var = mdl.var()
        exact_var = mdl.var(True)

        self.assertTrue(in_tol(sim_var, exact_var))


if __name__ == '__main__':
    unittest.main()
