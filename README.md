# Insurance Mathematics

Repository of mathematical models for use in insurance. 

## Aggregate Distributions

Aggregate distributions are defined as a base class. Underlying distributions call directly from the `scipy.stats` module. Methods are designed to as flexible as possible, with some parameters overrideable (such as frequency thinning).

### Fourier transform

1. Aggregate distribution with underlying Poisson frequency has closed form Fourier transform. Use the class [`poisson_fft_agg`](pkg/src/fft_poisson.py) from to parameterize suitable underlying Poisson distribution and specify a severity distribution.
    The Fourier transformed aggregate distribution with underlying Poisson parameter $\lambda$ is given by:
    $$\exp\Big( \lambda (\hat{p}_X - 1) \Big)$$
2. Similar arguments for the negative binomial frequency. Use the class [`nb_fft_agg`](pkg/src/fft_nb.py), specify the frequency parameters and specify a severity distribution.
    The Fourier transformed aggregate distribution with underlying negative binomial mean $\lambda$ and coefficent of variation (variance-to-mean ratio) $\nu$ is given by:
    $$\left(\frac{1}{1+(\nu-1)(1-\hat{p}_X)}\right)^{\frac{\lambda}{\nu-1}}$$

where $\hat{p}_X$ is the Fourier transformed severity PDF.

#### Frequency thinning

Both the Poisson and negative binomial distributions allow for "thinning". This is useful for determining excess of loss claim frequency. 

For a $\text{Poi}(\lambda)$ distribution, thinning with factor $\alpha$ yields another Poisson distribution:

$$\text{Poi}(\alpha \lambda)$$

For a $\text{NB}(n, p)$ distribution, where the parameters $n$ represents the number of successes within a trial and $p$ represents the probability of failure, thinning with factor $\alpha$ yields another Binomial distribution:

$$\text{NB}\left(n, \frac{\alpha p}{1-p+\alpha p} \right)$$

## Simulations

In the [`agg_sim`](pkg/src/agg_sim.py) module, aggregate distributions are created through simulation. The basic structure is:

- Choose a frequency distribution and simulate claim counts for each policy year.
- Within each policy year, generate claims from a chosen severity distribution and sum up to get an aggregate loss for that year.
- Repeat the above to generate a large number of possible aggregate loss years.
- Collect descriptive statistics from the above results.

The advantage of simulation over Fourier transform method is the flexibility. Inherit from this class and overwrite the `_generate_severities` function to implement custom features. Also any frequency/severity distribution combination can be used, so long as they follow implementation procedures similar to that of `scipy.stats`.

However, simulation techniques are slower and less accurate than that of the Fourier transform. To remedy the speed, the `compile_aggregate_distribution` function parallelizes calls to the `_generate_severities` procedure.

## TODOs:

- Implement/inherit from fft sub-classes for custom contracts (e.g. stop-loss program with dropdown post AAD/AAL break)

# Sources

1. Goelden, Heinz-Willi; Hess, Klaus Th.; Morlock, Martin; Schmidt, Klaus D; Schröter, Klaus J. - *Schadenversicherungsmathematik* - Springe Spektrum
2. Parodi, Pietro - *Pricing in General Insurance* - CRC Press LLC
