# import numpy as np

from ins_mat.agg_dist.fft_nb import nb_fft_agg
from scipy.stats import genpareto
import pandas as pd
import copy


def gross_nb_fft_test():
    print('Testing gross insurance structure with following:')
    f_np = [1.2, 0.5]
    x_gpd = {
            'dist': genpareto,
            'properties': [
                0.378,  # c
                0.000,  # loc
                8.123  # scale
                ]
            }

    print(f'Frequency: {f_np}')
    print('Severity:')
    print(x_gpd)
    print('\n')

    # Create instance
    gross_agg = nb_fft_agg(
            n=f_np[0],
            p=f_np[1],
            severity_distribution=x_gpd
            )

    # Compile the aggregate distribution
    gross_agg.compile_aggregate_distribution()

    # Inspect the validation by calling diagnostics
    print('Diagnostics:')
    print(gross_agg.diagnostics)
    print(f'PPF@90% = {gross_agg.ppf(0.9)}')
    print('\n')
    return gross_agg


def limited_nb_fft_test():
    print('Testing limited insurance structure with following:')
    f_np = [1.2, 0.5]
    x_gpd = {
            'dist': genpareto,
            'properties': [
                0.378,  # c
                0.000,  # loc
                8.123  # scale
                ]
            }

    print(f'Frequency: {f_np}')
    print('Severity:')
    print(x_gpd)
    print('\n')

    # Create instance
    lim_agg = nb_fft_agg(
            n=f_np[0],
            p=f_np[1],
            severity_distribution=x_gpd
            )

    # Set up limit and excess amounts
    xs, lim = 1.0, 5.0
    print(f'Excess = {xs:,.3f}')
    print(f'Limit = {lim:,.3f}')
    lim_agg.setup_layer(excess=xs, limit=lim)

    # Compile the aggregate distribution
    lim_agg.compile_aggregate_distribution()

    # Inspect the validation by calling diagnostics
    print('Diagnostics:')
    print(lim_agg.diagnostics)
    print(f'PPF@90% = {lim_agg.ppf(0.9)}')
    print('\n')
    return lim_agg


def stop_loss_nb_test():
    print('Testing stop-loss with the following:')
    f_np = [1.2, 0.5]

    x_gpd = {
            'dist': genpareto,
            'properties': [
                0.378,  # c
                0.000,  # loc
                8.123  # scale
                ]
            }

    print(f'Frequency: {f_np}')
    print('Severity:')
    print(x_gpd)
    print('\n')

    # Create instance of gross
    gross_agg = nb_fft_agg(
            n=f_np[0],
            p=f_np[1],
            severity_distribution=x_gpd
            )

    # Compile aggregate distribution
    gross_agg.compile_aggregate_distribution()

    # Copy the object for the stop loss
    sl_agg = copy.deepcopy(gross_agg)
    sl_agg.setup_agg_layer(0, None)

    # Produce summary stats
    results = pd.DataFrame(
            data={
                'Mean_loss': [gross_agg.mean(False), sl_agg.mean(False)],
                'Var_loss': [gross_agg.var(False), sl_agg.var(False)],
                '90percentile': [gross_agg.ppf(0.9), sl_agg.ppf(0.9)],
                '95percentile': [gross_agg.ppf(0.95), sl_agg.ppf(0.95)],
                },
            index=['Gross', '95%-Retained-Stop-Loss']
            )

    print(results)
    return [gross_agg, sl_agg]
