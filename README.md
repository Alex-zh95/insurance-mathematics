# Insurance Mathematics

Repository of mathematical models for use in insurance. 

## Aggregate Distributions

Aggregate distributions are defined as a base class. Underlying distributions call directly from the `scipy.stats` module. Methods are designed to as flexible as possible, with some parameters overrideable (such as frequency thinning).

### Fourier transform

1. Aggregate distribution with underlying Poisson frequency has closed form Fourier transform. Use the class [`poisson_fft_agg`](pkg/src/fft_poisson.py) from to parameterize suitable underlying Poisson distribution and specify a severity distribution.
    The Fourier transformed aggregate distribution with underlying Poisson parameter $\lambda$ is given by:
    $$\exp\Big( \lambda (\hat(p)_X - 1) \Big)$$

    where $\hat(p)_X$ is the Fourier transformed severity PDF.
2. (TODO): Similar arguments for the negative binomial frequency.

### Frequency thinning

Both the Poisson and negative binomial distributions allow for "thinning". This is useful for determining excess of loss claim frequency. 

For a $\text{Poi}(\lambda)$ distribution, thinning with factor $n$ yields another Poisson distribution:

$$\text{Poi}(n\lambda)$$

For a $\text{NB}(r, p)$ distribution, thinning with factor $n$ yields another Binomial distribution:

$$\text{NB}\left(r, \frac{np}{1-p+np} \right)$$

# Sources
1. Goelden, Heinz-Willi; Hess, Klaus Th.; Morlock, Martin; Schmidt, Klaus D; Schröter, Klaus J. - *Schadenversicherungsmathematik* - Springe Spektrum
2. Parodi, Pietro - *Pricing in General Insurance*
