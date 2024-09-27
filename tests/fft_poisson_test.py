from context import access_root_dir
access_root_dir(1)

from ins_mat.agg_dist.fft_poisson import Agg_PoiFft
from scipy.stats import genpareto


def gross_poisson_fft_test():
    print('Testing gross insurance structure with following:')
    f_lambda = 1.2
    x_gpd = {
            'dist': genpareto,
            'properties': [
                0.378,  # c
                0.000,  # loc
                8.123  # scale
                ]
            }

    print(f'Frequency mean = {f_lambda:,.5f}')
    print('Severity:')
    print(x_gpd)
    print('\n')

    # Create instance
    gross_agg = Agg_PoiFft(
            frequency=f_lambda,
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


def limited_poisson_fft_test():
    print('Testing layered insurance structure with following:')
    f_lambda = 1.2
    x_gpd = {
            'dist': genpareto,
            'properties': [
                0.378,  # c
                0.000,  # loc
                8.123  # scale
                ]
            }

    print(f'Frequency mean = {f_lambda:,.5f}')
    print('Severity:')
    print(x_gpd)
    print('\n')

    # Create instance
    agg = Agg_PoiFft(
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
    print('No diagnostics can be made for net-of-excess but we can check following:')
    print(f'Mean = {agg.mean(theoretical=False)}')
    print(f'Var = {agg.var(theoretical=False)}')
    print(f'PPF@90% = {agg.ppf(0.9)}')
    print('\n')

    return agg
