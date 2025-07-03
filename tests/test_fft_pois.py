import os
import sys

# If not locally installed (or at least in editable mode), append path the base dir and src dir for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from insurance_mathematics.agg_dist.fft_poisson import Agg_PoiFft
from scipy import stats

import unittest

from tests.fft_nb_test import percentage_error


class Test_FFT_Pois(unittest.TestCase):
    frequency: dict
    severity: dict
    layer: list

    @classmethod
    def setUpClass(cls) -> None:
        cls.frequency = {
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
        lambd = self.frequency['properties'][0]
        mdl = Agg_PoiFft(frequency=lambd, severity_distribution=self.severity)
        mdl.compile_aggregate_distribution()

        fft_mean = mdl.mean()
        exact_mean = mdl.mean(True)

        self.assertAlmostEqual(percentage_error(fft_mean, exact_mean), 0.0, 2)

    def test_layer(self):
        '''Testing consistency of calculations in layer between fft and exact.'''
        lambd = self.frequency['properties'][0]
        mdl = Agg_PoiFft(frequency=lambd, severity_distribution=self.severity)

        mdl.setup_layer(*self.layer)
        mdl.compile_aggregate_distribution()

        fft_mean = mdl.mean()
        exact_mean = mdl.mean(True)

        self.assertAlmostEqual(percentage_error(fft_mean, exact_mean), 0.0, 2)

    def test_variance(self):
        '''Test consistency of calcs for variance between FFT and exact.'''
        lambd = self.frequency['properties'][0]
        mdl = Agg_PoiFft(frequency=lambd, severity_distribution=self.severity)

        mdl.setup_layer(*self.layer)
        mdl.compile_aggregate_distribution()

        fft_var = mdl.var()
        exact_var = mdl.var(True)

        self.assertAlmostEqual(percentage_error(fft_var, exact_var), 0.0, 2)


if __name__ == '__main__':
    unittest.main()
