'''
Running various tests
'''

from pkg.tests import fft_poisson_test
from pkg.tests import fft_nb_test

print('Running FFT Poisson tests...\n')
fft_poisson_test.gross_poisson_fft_test()
fft_poisson_test.limited_poisson_fft_test()

print('Running FFT NB tests...\n')
fft_nb_test.gross_nb_fft_test()
fft_nb_test.limited_nb_fft_test()

print('Looking at Stop-Losses...\n')
fft_nb_test.stop_loss_nb_test()
