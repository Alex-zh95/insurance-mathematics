import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from insurance_mathematics.agg_dist.fft_poisson import Agg_PoiFft
from insurance_mathematics.stochastic.jump_diffusion import MertonJump_CompoundPoisson

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

# Set up the jump distribution
jump_mdl = Agg_PoiFft(frequency=pois_freq, severity_distribution=severity)
jump_mdl.compile_aggregate_distribution()
print(f'Fourier mean for jump:      {jump_mdl.mean():,.3f}')

# Plot probability against lr
lrs = np.linspace(0.4, 1.2, 100)
delts = []
pExers = []
pCall = []
pExers2 = []
pCall2 = []

for lr in lrs:
    main_mdl = MertonJump_CompoundPoisson(jump_mdl, 0.03, 0.15, lr)

    delts.append(main_mdl.pi1())
    pExers.append(main_mdl.pi2())
    pCall.append(main_mdl.price())

for lr in lrs:
    main_mdl = MertonJump_CompoundPoisson(jump_mdl, 0.03, 0.25, lr)
    pExers2.append(main_mdl.pi2())
    pCall2.append(main_mdl.price())

# Plot graph
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 8)
ax[0].plot(lrs, pExers, label='Sig=0.15')
ax[0].plot(lrs, pExers2, label='Sig=0.25')
ax[0].set_xlabel('Loss ratio')
ax[0].set_ylabel('Exercise probability')
ax[0].grid()
ax[1].plot(lrs, pCall, label='Sig=0.15')
ax[1].plot(lrs, pCall2, label='Sig=0.25')
ax[1].set_xlabel('Loss ratio')
ax[1].set_ylabel('Call price')
ax[1].grid()
ax[1].legend()
fig.show()
