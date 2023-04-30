'''
Running various tests
'''

from pkg.tests import fft_poisson_test

print('Running FFT Poisson tests...\n')
fft_poisson_test.gross_poisson_fft_test()
fft_poisson_test.limited_poisson_fft_test()
