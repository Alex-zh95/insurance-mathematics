from insurance_mathematics.agg_dist.fft_poisson import Agg_PoiFft
from insurance_mathematics.agg_dist.agg_sim import AggSim
from scipy import stats
import numpy as np

# Params
pois_freq = 0.5

frequency = {
    'dist': stats.poisson,
    'properties': [pois_freq]
}

severity = {
    'dist': stats.lognorm,
    'properties': [1.2, 0.0, 7.8]
}


def perc_err(observed: float, expected: float) -> float:
    return np.abs(observed - expected) / expected


def ground_up_test():
    print("Ground-up test...")
    mdl_fs = Agg_PoiFft(frequency=pois_freq, severity_distribution=severity)
    mdl_sim = AggSim(frequency, severity)

    mdl_fs.compile_aggregate_distribution()
    mdl_sim.compile_aggregate_distribution()

    print(f'Fourier mean:       {mdl_fs.mean()}')
    print(f'Simulated mean:     {mdl_sim.mean()}')
    print(f'Exact mean:         {mdl_fs.mean(True)}')
    print('\n')

    if perc_err(mdl_fs.mean(), mdl_fs.mean(True)) < 1e-3:
        return 0
    else:
        return 1


def layer_test():
    print("Layer test...")

    mdl_fs = Agg_PoiFft(frequency=pois_freq, severity_distribution=severity)
    mdl_sim = AggSim(frequency, severity)

    mdl_fs.setup_layer(excess=3.0, limit=10.0)
    mdl_sim.setup_layer(excess=3.0, limit=10.0)

    mdl_fs.compile_aggregate_distribution()
    mdl_sim.compile_aggregate_distribution()

    print(f'Fourier mean:       {mdl_fs.mean()}')
    print(f'Simulated mean:     {mdl_sim.mean()}')
    print(f'Exact mean:         {mdl_fs.mean(True)}')
    print('\n')

    if perc_err(mdl_fs.mean(), mdl_fs.mean(True)) < 1e-3:
        return 0
    else:
        return 1


def agg_lim_test():
    print("Agg lim test...")
    print("Note: no theoretical test possible - compare sim and fft")
    print("Sims have greater error chances, so bound check increased")

    mdl_fs = Agg_PoiFft(frequency=pois_freq, severity_distribution=severity)
    mdl_sim = AggSim(frequency, severity)

    mdl_fs.setup_layer(excess=5.0, limit=5.0)
    mdl_fs.setup_agg_limit(agg_limit=8.5)
    mdl_sim.setup_layer(excess=5.0, limit=5.0, agg_limit=8.5)

    mdl_fs.compile_aggregate_distribution()
    mdl_sim.compile_aggregate_distribution()

    print(f'Fourier mean:       {mdl_fs.mean()}')
    print(f'Simulated mean:     {mdl_sim.mean()}')
    print('\n')

    if perc_err(mdl_fs.mean(), mdl_sim.mean()) < 0.1:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit_codes = 0
    exit_codes += ground_up_test()
    exit_codes += layer_test()
    exit_codes += agg_lim_test()

    if exit_codes <= 1:
        print("All tests passed.")
