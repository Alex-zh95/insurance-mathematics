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

### Frequency thinning

Both the Poisson and negative binomial distributions allow for "thinning". This is useful for determining excess of loss claim frequency. 

For a $\text{Poi}(\lambda)$ distribution, thinning with factor $\alpha$ yields another Poisson distribution:

$$\text{Poi}(\alpha \lambda)$$

For a $\text{NB}(n, p)$ distribution, thinning with factor $\alpha$ yields another Binomial distribution:

$$\text{NB}\left(n, \frac{\alpha p}{1-p+\alpha p} \right)$$

## TODOs:

- Implement/inherit from fft sub-classes for custom contracts (e.g. stop-loss program with dropdown post AAD/AAL break)
- Inherit from aggregate base class methods for simulation techniques (will be useful for tests)

# Sources

1. Goelden, Heinz-Willi; Hess, Klaus Th.; Morlock, Martin; Schmidt, Klaus D; Schröter, Klaus J. - *Schadenversicherungsmathematik* - Springe Spektrum
2. Parodi, Pietro - *Pricing in General Insurance* - CRC Press LLC
