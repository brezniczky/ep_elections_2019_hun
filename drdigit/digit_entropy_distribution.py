from scipy.stats import entropy, poisson
import numpy as np
from functools import lru_cache
import pandas as pd
from joblib import Memory


_DEFAULT_SEED = 1234
_DEFAULT_ITERATIONS = 10000


_DEFAULT_BOTTOM_N = 20
_DEFAULT_LL_SEED = 1234
_DEFAULT_LL_ITERATIONS = 1000
_DEFAULT_PE_RANDOM_SEED = 1234
_DEFAULT_PE_ITERATIONS = 12345


mem = Memory("./.digit_entropy_distribution_cache", verbose=0)


def get_entropy(x):
    """ x should be a list-like of digits """
    _, counts = np.unique(x, return_counts=True)
    counts = counts / sum(counts)
    # no need to pad for missing digits, terms with a zero coeff.
    # fall out anyway
    #
    # if len(counts) < 10:
    #     counts = np.concatenate([counts, [0] * (10 - len(counts))], axis=0)
    return entropy(counts)


# this one gives a very uneven distribution
#
# plt.hist(abs(np.diff(np.random.choice(10, 10000))), bins=range(11))
# plt.show()
#
# def get_abs_diff_entropy(x):
#     """ x should be a list-like of digits """
#     return abs(get_entropy(np.diff(x)))
#
# could be sensitive to changes in certain normally rare (e.g. 9)
# differences by the look


def prob_of_twins(x):
    """ Return the probability of at least this many repeats
        in the sequence of digits

        x: list-like of digits
    """
    if len(x) <= 1:
        # nothing to see here, not enough information
        return 1
    x = np.array(x)
    count = sum(x[:-1] == x[1:])
    return 1 - poisson((len(x) - 1) / 10).cdf(count - 1)


@mem.cache()
def generate_sample(n_wards,
                    seed=_DEFAULT_SEED,
                    iterations=_DEFAULT_ITERATIONS,
                    quiet=False):
    np.random.seed(seed)
    entrs = []
    for i in range(iterations):
        values = np.random.choice(range(10), n_wards)
        _, counts = np.unique(values, return_counts=True)
        entr = entropy(counts / sum(counts))
        entrs.append(entr)
    if not quiet:
        print("cdf for %d was generated" % n_wards)
    return np.array(entrs)


@lru_cache(1000)
def get_cdf_fun(n_wards,
                seed=_DEFAULT_SEED,
                iterations=_DEFAULT_ITERATIONS,
                quiet=False):
    """ Return a PMF of digit distribution entropy values, that is, F where
        F(y) = P(X <= y)

        The entropy values considered are those returned by
        scipy.stats.entropy() for the relative digit frequency
        configuration. """
    sample = generate_sample(n_wards, seed, iterations, quiet=quiet)
    values, counts = np.unique(sample, return_counts=True)
    total = sum(counts)
    counts = np.cumsum(counts)

    def cdf_fun(y, avoid_inf=False):
        idx = np.digitize(y, values) - 1
        if idx >= 0:
            return counts[idx] / total
        elif not avoid_inf:
            return 0
        else:
            # give a probability that likely goes undetected
            return 1 / total

    return cdf_fun


def prob_of_entr(n_wards, entr,
                 seed=_DEFAULT_SEED,
                 iterations=_DEFAULT_ITERATIONS,
                 avoid_inf=False,
                 quiet=False):
    """ Probability of the entropy being this small. """
    cdf = get_cdf_fun(n_wards, seed, iterations, quiet)
    return cdf(entr, avoid_inf=avoid_inf)


"""
Log likelihood assessment of groups of digits of varying size for uniformity.

Uses simulations, slow.

Relies on non-parametric CDFs generated above.
"""
# TODO: towns argument should be generalized or municipalities
def get_log_likelihood(digits, slice_limits, bottom_n,
                       seed, iterations, towns=None, return_probs=False,
                       avoid_inf=False, quiet=False):
    slices = [digits[a:b] for a, b in slice_limits]
    entropies = [get_entropy(s) for s in slices]
    probs = [prob_of_entr(len(s), e, seed, iterations,
                          avoid_inf=avoid_inf, quiet=quiet)
             for s, e in zip(slices, entropies)]
    bottom_probs = sorted(probs)[:bottom_n]
    # TODO: remove the towns param that even I can't remember ;)
    if towns is not None and not quiet:
        print(towns[np.array(probs) <= max(bottom_probs)])
    l = sum(np.log(bottom_probs))
    if not return_probs:
        return l
    else:
        return l, probs


def get_likelihood_cdf(slice_limits, bottom_n,
                       seed=_DEFAULT_LL_SEED,
                       iterations=_DEFAULT_LL_ITERATIONS,  # voting simulation
                       pe_seed=_DEFAULT_PE_RANDOM_SEED,
                       pe_iterations=_DEFAULT_PE_ITERATIONS,
                       quiet=False):  # entropy

    sample = []
    # end of the last slice is the ...
    n_settlements = slice_limits[-1][1]
    warnings = 0

    """
    To allow the other function (ugly as hell) generate CDFs using
    its own random seed control, and later have exclusive control
    over the seed, ensure the CDFs are ready ahead of the iterations.
    (Temp. fix - should have different random sequences.)
    """
    np.random.seed(seed)
    for i in range(min(iterations, 1)):
        digits = np.random.choice(range(10), n_settlements)
        sim_likelihood = get_log_likelihood(digits, slice_limits,
                                            bottom_n, pe_seed, pe_iterations,
                                            quiet=quiet)

    """ Now carry out the actual calculations... """
    np.random.seed(seed)
    for i in range(iterations):
        digits = np.random.choice(range(10), n_settlements)
        sim_likelihood = get_log_likelihood(digits, slice_limits,
                                            bottom_n, pe_seed, pe_iterations,
                                            quiet=quiet)
        sample.append(sim_likelihood)
        # -Inf will count in a "forgiving" way still, so a warning only
        if np.isinf(sim_likelihood) and not quiet:
            print("Warning! Infinite simulated likelihood encountered - "
                  "perhaps increase the p.e. iterations?")
            warnings += 1
        if i % 50 == 0 and not quiet:
            print(i, np.mean(np.array(sample)[~np.isinf(sample)]))

    values, counts = np.unique(sample, return_counts=True)
    total = sum(counts)
    counts = np.cumsum(counts)

    def cdf(l):
        """ CDF: P(L <= L_actual) """
        idx = np.digitize(l, values) - 1
        if idx >= 0:
            return counts[idx] / total
        else:
            return 0
    print("There were", warnings, "warnings.")

    cdf.min = min(values[~np.isinf(values)])
    cdf.max = max(values[~np.isinf(values)])
    cdf.sample = sample

    return cdf


class LogLikelihoodDigitGroupEntropyTest():

    # TODO: align this is_quiet interpretation with the one from arguments.py

    @staticmethod
    def get_slice_limits(group_ids):
        assert all(group_ids == sorted(group_ids)), \
               "Settlement values must be in a sorted order!"
        values = np.unique(group_ids)
        settlement_index = {
            name: i for name, i in zip(values, range(len(values)))
        }
        indexes = pd.Series([settlement_index[g] for g in group_ids])
        prev_indexes = pd.Series([-1] + list(indexes[:-1]))
        next_indexes = pd.Series(list(indexes[1:]) + [len(indexes)])
        starts = np.where(indexes != prev_indexes)[0]
        ends = np.where(indexes != next_indexes)[0] + 1
        return list(zip(starts, ends))

    def __init__(self,
                 digits, group_ids, bottom_n,
                 # log likelihood probability calc. hyperparam.
                 ll_iterations=_DEFAULT_LL_ITERATIONS,
                 ll_seed=_DEFAULT_LL_SEED,
                 # probability of entropy hyperparam.
                 pe_iterations=_DEFAULT_PE_ITERATIONS,
                 pe_seed=_DEFAULT_SEED,
                 quiet=False):

        # avoid index keys interfering with the intentions
        if type(digits) is pd.Series:
            digits = digits.values
        if type(group_ids) is pd.Series:
            group_ids = group_ids.values

        # TODO: should I carry the sorting out here? yes
        df = pd.DataFrame({
            "group_id": group_ids,
            "digit": digits,
        })
        df.sort_values(["group_id"], inplace=True)
        self._digits = df["digit"].values
        self._group_ids = df["group_id"].values
        self._slice_limits = self.get_slice_limits(self._group_ids)

        self._bottom_n = bottom_n
        self._ll_iterations = ll_iterations
        self._ll_seed = ll_seed
        self._pe_iterations = pe_iterations
        self._pe_seed = pe_seed

        self._likelihood = None
        self._p_values = None
        self._cdf = None
        self._quiet = quiet

    def _init_likelihood_and_probs(self):
        self._likelihood, self._p_values = get_log_likelihood(
            self._digits, self._slice_limits,
            self._bottom_n,
            self._pe_seed, self._pe_iterations,
            return_probs=True,
            # obtain an "upper estimate" on the likelihood to
            # avoid possible Infs resulting from numeric resolution
            avoid_inf=True,
            quiet=self._quiet
        )

    @property
    def likelihood(self):
        if self._likelihood is None:
            self._init_likelihood_and_probs()
        return self._likelihood

    @property
    def cdf(self):

        if self._cdf is None:
            self._cdf = get_likelihood_cdf(
                self._slice_limits, self._bottom_n,
                self._ll_seed, self._ll_iterations,
                self._pe_seed, self._pe_iterations,
                quiet=self._quiet
            )
        return self._cdf  # TODO: that is not exactly read-only...

    @property
    def p(self):
        return self.cdf(self.likelihood)

    @property
    def p_values(self):
        if self._p_values is None:
            self._init_likelihood_and_probs()
        return dict(
            list(zip(self._group_ids, self._p_values))
        )


def test():
    res = list(LodigeTest.get_slice_limits([1, 2, 2, 3, 3, 3]))
    assert (
        res ==
            [(0, 1),
             (1, 3),
             (3, 6)]
    )


""" Short name for now... """
LodigeTest = LogLikelihoodDigitGroupEntropyTest


if __name__ == "__main__":
    # TODO: of course move this to a test suite someday
    test()

    # print(prob_of_entr(47, 2.1485))
    # expect ~ 0.133 (got with seed=1234) depending on the seed
    test = LodigeTest(
        [1] * 8 + [2] * 4, [1] * 12, 10
    )
    print(test.p)
    test = LodigeTest(
        list(range(10)), [10] * 10, 10
    )
    print(test.p)
    test = LodigeTest(
        list(range(5)) + [1] * 5, [10] * 10, 10
    )
    print(test.p)


"""
In [117]:
f2 = get_cdf_fun(200)
ys2 = np.array([f2(x) for x in xs])
xs2 = xs[ys2 < 0.5]
ys2 = ys2[ys2 < 0.5]
plt.plot(xs2, np.log(ys2))
plt.show()

f2 = get_cdf_fun(100)
ys2 = np.array([f2(x) for x in xs])
xs2 = xs[ys2 < 0.5]
ys2 = ys2[ys2 < 0.5]
plt.plot(xs2, np.log(ys2))
plt.show()

f2 = get_cdf_fun(500, iterations=50000)
ys2 = np.array([f2(x) for x in xs])
xs2 = xs[ys2 < 0.5]
ys2 = ys2[ys2 < 0.5]
plt.plot(xs2, np.log(ys2))
plt.show()


TODO: left tail apparently (and as conjectured from
early stage analytical work) could be approximated by
an exponential form.
Then we could estimate probability of outlying
extremities based on that.
"""
