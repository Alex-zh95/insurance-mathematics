'''
Running various tests
'''

from ins_mat.tests import fft_poisson_test
from ins_mat.tests import fft_nb_test
from ins_mat.tests import mc_test
from ins_mat.tests import credit_test

# print('Running FFT Poisson tests...\n')
# agg1 = fft_poisson_test.gross_poisson_fft_test()
# agg2 = fft_poisson_test.limited_poisson_fft_test()
#
# print('Running FFT NB tests...\n')
# agg3 = fft_nb_test.gross_nb_fft_test()
# agg4 = fft_nb_test.limited_nb_fft_test()
#
# print('Looking at Stop-Losses...\n')
# agg5 = fft_nb_test.stop_loss_nb_test()
#
# print('Running MC-Sim test')
# agg6 = mc_test.agg_lim_test()
#
print('Running credit rating test')
in_str = input('Ticker strings (separate by commas): ')
in_lim = input('Set limit: ')
names = in_str.split(', ')
credit_test.main_test(names, float(in_lim))
