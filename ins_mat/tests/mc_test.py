from ins_mat.agg_dist.agg_sim import agg_sim
from ins_mat.agg_dist.fft_nb import nb_fft_agg
from ins_mat.agg_dist.fft_poisson import poisson_fft_agg
from scipy.stats import genpareto, nbinom, poisson


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
    print(f'PPF@90% = {gross_agg.ppf(0.9)}')
    print('\n')

    # Now apply the simulations
    gross_sim = agg_sim(
            frequency_distribution={'dist': nbinom, 'properties': f_np},
            severity_distribution=x_gpd,
            )

    # Simlate aggregate distribution
    gross_sim.compile_aggregate_distribution()

    print('Simulation diagnostics:')
    print(f'Aggregate mean: {gross_sim.mean(False):.5f}')
    print(f'Aggregate var: {gross_sim.var(False):.5f}')
    print(f'PPF@90% = {gross_sim.ppf(0.9)}')

    return gross_sim


def agg_lim_test():
    print('Testing simulation results against FFT and theoretical results')
    f_lambda = 1.2
    x_gpd = {
            'dist': genpareto,
            'properties': [
                0.378,  # c
                0.000,  # loc
                8.123  # scale
                ]
            }

    print(f'Frequency: {f_lambda}')
    print('Severity:')
    print(x_gpd)
    print('\n')

    # Create instance of the FFT
    agg = poisson_fft_agg(
            frequency=f_lambda,
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
    print('Diagnostics:')
    print(agg.diagnostics)
    print(f'PPF@90% = {agg.ppf(0.9)}')
    print('\n')

    # Now apply the simulations
    lim_sim = agg_sim(
            frequency_distribution={'dist': poisson, 'properties': [f_lambda]},
            severity_distribution=x_gpd,
            excess=xs,
            limit=lim,
            n=int(1e6)
            )

    lim_sim.compile_aggregate_distribution()

    print('Simulation diagnostics:')
    print(f'Aggregate mean: {lim_sim.mean("False"):.5f}')
    print(f'Aggregate var: {lim_sim.var("False"):.5f}')
    print(f'PPF@90% = {lim_sim.ppf(0.9)}')

    return lim_sim
