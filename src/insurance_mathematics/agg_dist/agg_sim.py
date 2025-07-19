import numpy as np
import bisect
from joblib import Parallel, delayed

from insurance_mathematics.agg_dist.agg_base import AggregateDistribution


class AggSim(AggregateDistribution):
    """
    Define and build aggregate distribution via Monte-Carlo simulation from given frequency and severity distributions.

    Inherits: aggregate_dstribution

    Distributions take the form:

    {
        dist: scipy.stats
        properties: statistical parameters required (in dict form)
    }

    Parameters
    ----------
    frequency_distribution: as described above
    severity_distribution: as described above

    n: int, default = 100,000
        Number of simulations to run
    excess: float, default = 0
        Set the each and every excess
    limit: float | None, default = None
        Set the each and every limit (None for infinite)
    aag_excess: float, default = 0
        Set the aggregate deductible
    agg_limit: float | None, default = None
        Set the aggregate limit (None for infinite)

    Callables
    ---------
    Some basic stats can be called for the underlying frequency and severity distributions. These are mostly calls to their scipy.stats functions. Refer to scipy manual for more details.

    Call compile_aggregate_distribution() to generate the aggregate distribution.
    """

    n_sims: int
    xs: float
    lim: float
    agg_d: float
    agg_l: float
    parallel: bool

    def __init__(
        self,
        frequency_distribution: dict,
        severity_distribution: dict,
        n: int = 100000,
        excess: float = 0,
        limit: float | None = None,
        agg_excess: float = 0,
        agg_limit: float | None = None,
        parallel: bool = True,
    ):
        super().__init__(frequency_distribution, severity_distribution)
        self.n_sims = n

        self.xs = excess
        self.lim = limit
        self.agg_d = agg_excess
        self.agg_l = agg_limit
        self.parallel = parallel

    def _generate_severities(self, freq):
        """
        Internal function to generate the severities, including modifications to excess and limit if needed (None when not).
        """
        severities = self.severity["dist"].rvs(*self.severity["properties"], size=freq)

        severities = np.fmax(severities - self.xs, 0)

        if self.lim is not None:
            severities = np.fmin(severities, self.lim)

        return (
            min(max(severities.sum() - self.agg_d, 0), self.agg_l)
            if self.agg_l is not None
            else max(severities.sum() - self.agg_d, 0)
        )

    def compile_aggregate_distribution(self):
        """
        Generate aggregate PDF and CDF via Monte-Carlo simulation.

        Via frequency distribution, generate a random number of severities and for each severity, via the severity distribution, generate a random loss.

        Generate the aggregate loss simulation by adding all severities.
        """

        n_losses = self.frequency["dist"].rvs(
            *self.frequency["properties"], size=self.n_sims
        )

        parallel_pool = Parallel(n_jobs=-1 if self.parallel else 1)
        delayed_generate_severities = (
            delayed(self._generate_severities)(n) for n in n_losses
        )

        self.losses = np.sort(parallel_pool(delayed_generate_severities))

        if (
            (self.lim is None)
            & (self.xs is None)
            & (self.agg_d is None)
            & (self.agg_l is None)
        ):
            self._validate()

    # Aggregate statistics overrides
    def mean(self, theoretical: bool = False):
        """
        Returns the mean of the aggregate distribution.

        If `theoretical` is set to true and ground up, we return

        E(severity)*E(frequency)

        Otherwise return the simulated mean
        """
        if (theoretical) & (self.lim is None) & (self.xs == 0):
            return self.get_severity_mean() * self.get_frequency_mean()
        else:
            return np.mean(self.losses)

    def var(self, theoretical: bool = False):
        """
        Returns the variance of the aggregate distribution. If `theoretical` is set to true, we return

        E(frequency)*V(severity) + V(frequency)*E(severity)^2

        Otherwise return the simulated variance
        """
        if (theoretical) & (self.lim is None) & (self.xs == 0):
            return (
                self.get_frequency_mean() * self.get_severity_variance()
                + self.get_frequency_variance() * (self.get_severity_mean()) ** 2
            )
        else:
            return np.var(self.losses, ddof=1)

    def ppf(self, q: float | list):
        """
        Return the percentage point of aggregate.

        Parameters
        ----------
        q: float | list
            Percentile (between 0 and 1)

        Returns
        -------
        result: np.ndarray
            Corresponding percentage point on aggregate distribution
        """
        return np.quantile(self.losses, q)

    def cdf(self, x: float | list):
        """
        Return the cdf of aggregate.

        Parameters
        ----------
        x: float
            Loss amount (must be in range of loss vector)

        Returns
        -------
        result: np.ndarray
            Corresponding cdf on aggregate distribution
        """
        _x = np.array([x]) if isinstance(x, float) else np.array(x)

        assert ((x >= self.losses.min()).all()) & (
            (x <= self.losses.max()).all()
        ), "loss x out of scope"

        # Obtain empirical pdf by using obtaining relative position in vector
        indices = np.array([bisect.bisect(self.losses, xi) for xi in _x])
        agg_cdf = indices / self.losses.shape[0]

        # Pass corresponding pdf output
        return agg_cdf

    def setup_layer(
        self,
        excess: float,
        limit: float | None,
        agg_excess: float = 0,
        agg_limit: float | None = None,
        inplace: bool = True,
    ):
        """
        Set up a suitable insurance structure with each-and-every excess and limits.

        Layer designed by adjusting probability of incurring losses after excess and limit.

        Parameters
        ----------
        excess: float,
            Excess retained (by insured)
        limit: float
            Limit, above which losses are ceded. Use None for no limit.
        inplace: bool [Optional]
            If set, we overwrite the current discretized severity PDF, otherwise, return the discretized PDF as a vector, which can be inserted into another object

        Returns
        -------
        new_sim_agg: agg_sim | None
            agg_sim object with new parameters (only when inplace is not set to True)
        """
        if inplace:
            self.xs = excess
            self.lim = limit
            self.agg_d = agg_excess
            self.agg_l = agg_limit

            return None
        else:
            new_sim_agg = AggSim(
                frequency_distribution=self.frequency,
                severity_distribution=self.severity,
                n=self.nsims,
                excess=excess,
                limit=limit,
                agg_excess=agg_excess,
                agg_limit=agg_limit,
            )
            return new_sim_agg

    def setup_agg_limit(self, agg_limit: float = 0.0, inplace: bool = True):
        """Somewhat redundant - but maintains inheritance conventions - uses setup_layer(...)"""
        self.setup_layer(
            excess=self.excess,
            limit=self.limit,
            agg_excess=0.0,
            agg_limit=agg_limit,
            inplace=inplace,
        )

    def pdf(self, x: float | list):
        """This function not needed for simulation"""
        pass

    def thin_frequency(self, n: float):
        """Function would be frequency-distribution-specific. Not implemented here to maintain flexibility for the purpose of simulation

        Consider implmenting outside of this or inheriting from agg_sim if thinning is desired.
        """
        pass
