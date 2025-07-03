# Insurance Mathematics

Repository of mathematical models for use in insurance. 

## Aggregate Distributions

Aggregate distributions are defined as a base class. Underlying distributions call directly from the `scipy.stats` module. Methods are designed to as flexible as possible, with some parameters overrideable (such as frequency thinning).

### Fourier transform

1. Aggregate distribution with underlying Poisson frequency has closed form Fourier transform. Use the class [`Agg_PoiFft`](pkg/src/fft_poisson.py) from to parameterize suitable underlying Poisson distribution and specify a severity distribution.
    The Fourier transformed aggregate distribution with underlying Poisson parameter $\lambda$ is given by:

    $$\exp\Big( \lambda (\hat{p}_X - 1) \Big)$$

2. Similar arguments for the negative binomial frequency. Use the class [`Agg_NbFft`](pkg/src/fft_nb.py), specify the frequency parameters and specify a severity distribution.

    The Fourier transformed aggregate distribution with underlying negative binomial mean $\lambda$ and coefficent of variation (variance-to-mean ratio) $\nu$ is given by:

    $$\left(\frac{1}{1+(\nu-1)(1-\hat{p}_X)}\right)^{\frac{\lambda}{\nu-1}}$$

    where $\hat{p}_X$ is the Fourier transformed severity PDF.

#### Frequency thinning

Both the Poisson and negative binomial distributions allow for "thinning". This is useful for determining excess of loss claim frequency. 

For a $\text{Poi}(\lambda)$ distribution, thinning with factor $\alpha$ yields another Poisson distribution:

$$\text{Poi}(\alpha \lambda)$$

For a $\text{NB}(n, p)$ distribution, where the parameters $n$ represents the number of successes within a trial and $p$ represents the probability of failure, thinning with factor $\alpha$ yields another Binomial distribution:

$$\text{NB}\left(n, \frac{\alpha p}{1-p+\alpha p} \right)$$

Additional advantages of using the Fourier transform is that it generates a characteristic function, which can be much easier to modify when it comes to algebra of random variables. This is motivates the creation of a simplified capital model that can be constructed from financial documents for the purposes of reinsurance, described in more detail [below](#further-stochastic-models).

### Simulations

In the [`AggSim`](pkg/src/agg_sim.py) module, aggregate distributions are created through simulation. The basic structure is:

- Choose a frequency distribution and simulate claim counts for each policy year.
- Within each policy year, generate claims from a chosen severity distribution and sum up to get an aggregate loss for that year.
- Repeat the above to generate a large number of possible aggregate loss years.
- Collect descriptive statistics from the above results.

The advantage of simulation over Fourier transform method is the flexibility. Inherit from this class and overwrite the `_generate_severities` function to implement custom features. Also any frequency/severity distribution combination can be used, so long as they follow implementation procedures similar to that of `scipy.stats`.

However, simulation techniques are slower and less accurate than that of the Fourier transform. To remedy the speed, the `compile_aggregate_distribution` function parallelizes calls to the `_generate_severities` procedure.

# Further stochastic models

One area of branching out is the development of a simplified pricing capital model for the purpose of evaluating experience of insurance companies' individual lines of business where information is available. The motivation begins by considering a structural credit model, adapted to fit the insurance capital against its liabilities, where options pricing theory can be adopted.

## Merton jump diffusion model

This is an extension of the Geometric Brownian Motion (GBM) for asset prices. We allow for large losses as jumps in the asset valuation at discrete (but unknown) times. An implementation is available in `jump_diffusion.py`, which derives jumps via a the `Agg_PoiFft` class, which can be used to value:

- probability of insurance default (i.e. requiring reinsurer payout), modeled as the non-probability of exercise of the equivalent call option,
- credit spread of an equivalent insurance-linked security against risk-free yields.

The solution of the jump diffusion model uses the Carr-Madan formulas, which are a modified Fourier inversion formula.

# Sources

1. Goelden, Heinz-Willi; Hess, Klaus Th.; Morlock, Martin; Schmidt, Klaus D; Schröter, Klaus J. - *Schadenversicherungsmathematik* - Springe Spektrum
2. Parodi, Pietro - *Pricing in General Insurance* - CRC Press LLC
3. Robertson, John P - [*The Computation of Aggregate Loss Distributions*](https://www.casact.org/sites/default/files/2021-02/pubs_proceed_proceed92_92057.pdf) - Casualty Actuary Society
4. Carr, Madan: - *Option valuation using the Fast Fourier Transform*.
5. Merton - *Option pricing when underlying stock returns are discontinuous*
