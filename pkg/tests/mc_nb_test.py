from pkg.src.agg_sim import agg_sim
from pkg.src.fft_nb import nb_fft_agg
from scipy.stats import genpareto, nbinom
import pandas as pd

def agg_test():
    print('Testing simulation results against FFT and theoretical results')
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

    # Create instance of the FFT
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
    print(f'PPF@90% = {gross_agg.agg_ppf(0.9)}')
    print('\n')

    # Now apply the simulations
    gross_sim = agg_sim(
            frequency_distribution = {'dist': nbinom, 'properties': f_np},
            severity_distribution = x_gpd,
            )

    # Simlate aggregate distribution
    gross_sim.compile_aggregate_distribution()

    print('Simulation diagnostics:')
    print(f'Aggregate mean: {gross_sim.agg_mean(False):.5f}')
    print(f'Aggregate var: {gross_sim.agg_variance(False):.5f}')
    print(f'PPF@90% = {gross_sim.agg_ppf(0.9)}')

    return gross_sim

def agg_lim_test():
    print('Testing simulation results against FFT and theoretical results')
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

    # Create instance of the FFT
    agg = nb_fft_agg(
            n=f_np[0],
            p=f_np[1],
            severity_distribution=x_gpd
            )

    # Set up limit and excess amounts
    xs, lim = 1.0, 5.0
    print(f'Excess = {xs:,.3f}')
    print(f'Limit = {lim:,.3f}')
    agg.setup_layer(excess=xs, limit=lim)

    print(f'Excess frequency mean: {agg.get_frequency_mean():,.3f}')
    print('Excess severity:')
    print(x_gpd)

    # Compile the aggregate distribution
    agg.compile_aggregate_distribution()

    # Inspect the validation by calling diagnostics
    print('No diagnostics can be made for net-of-excess but we can check following:')
    print(f'Mean = {agg.agg_mean(theoretical=False)}')
    print(f'Var = {agg.agg_variance(theoretical=False)}')
    print(f'PPF@90% = {agg.agg_ppf(0.9)}')
    print('\n')

    # Now apply the simulations
    lim_sim = agg_sim(
            frequency_distribution = {'dist': nbinom, 'properties': f_np},
            severity_distribution = x_gpd,
            excess=xs,
            limit=lim
            )

    # We need the thinning of frequency (not implemented in class)
    # lim_sim.compile_aggregate_distribution()
    #
    # p = lim_sim.frequency['properties'][1]
    # # k is excess survival
    # k = 1-lim_sim.get_agg_cdf(lim_sim.xs)
    # lim_sim.frequency['properties'][1] = k*p/(1-p+k*p)

    # Simlate aggregate distribution - yes we'd have to do this twice
    lim_sim.compile_aggregate_distribution()

    print('Simulation diagnostics:')
    print(f'Aggregate mean: {lim_sim.agg_mean(False):.5f}')
    print(f'Aggregate var: {lim_sim.agg_variance(False):.5f}')
    print(f'PPF@90% = {lim_sim.agg_ppf(0.9)}')

    return lim_sim
