from scipy.stats import entropy, poisson
import numpy as np
from functools import lru_cache


_DEFAULT_SEED = 1234
_DEFAULT_ITERATIONS = 10000


_DEFAULT_BOTTOM_N = 20
_DEFAULT_RANDOM_SEED = 1234
_DEFAULT_ITERATIONS = 10
_DEFAULT_PE_RANDOM_SEED = 1234
_DEFAULT_PE_ITERATIONS = 1234  # 12345


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


def generate_sample(n_wards,
                    seed=_DEFAULT_SEED,
                    iterations=_DEFAULT_ITERATIONS):
    np.random.seed(seed)
    entrs = []
    for i in range(iterations):
        values = np.random.choice(range(10), n_wards)
        _, counts = np.unique(values, return_counts=True)
        entr = entropy(counts / sum(counts))
        entrs.append(entr)
    print("cdf for %d was generated" % n_wards)
    return np.array(entrs)


@lru_cache(maxsize=1000)
def get_cdf_fun(n_wards,
                seed=_DEFAULT_SEED,
                iterations=_DEFAULT_ITERATIONS):
    """ Return a 'forgiving' CDF, that is, one where
        equality allowed: F where
        F(y) = P(X <= y) """
    sample = generate_sample(n_wards, seed, iterations)
    values, counts = np.unique(sample, return_counts=True)
    total = sum(counts)
    counts = np.cumsum(counts)

    def cdf_fun(y):
        idx = np.digitize(y, values) - 1
        if idx >= 0:
            return counts[idx] / total
        else:
            return 0

    return cdf_fun


def prob_of_entr(n_wards, entr,
                 seed=_DEFAULT_SEED,
                 iterations=_DEFAULT_ITERATIONS):
    """ probability of the entropy being this small """
    cdf = get_cdf_fun(n_wards, seed, iterations)
    return cdf(entr)


"""
Log likelihood assessment of groups of digits of varying size for uniformity.

Uses simulations, slow.

Relies on non-parametric CDFs.
"""
def get_log_likelihood(digits, slice_limits, bottom_n,
                       seed, iterations, towns=None):
    slices = [digits[a:b] for a, b in slice_limits]
    entropies = [get_entropy(s) for s in slices]
    probs = [prob_of_entr(len(s), e, seed, iterations)
            for s, e in zip(slices, entropies)]
    bottom_probs = sorted(probs)[:bottom_n]
    if towns is not None:
        print(towns[np.array(probs) <= max(bottom_probs)])
    l = sum(np.log(bottom_probs))
    return l


def get_likelihood_cdf(slice_limits, bottom_n,
                       seed=_DEFAULT_RANDOM_SEED,
                       iterations=_DEFAULT_ITERATIONS,  # voting simulation
                       pe_seed=_DEFAULT_RANDOM_SEED,
                       pe_iterations=_DEFAULT_ITERATIONS):  # entropy

    sample = []
    # end of the last slice is the ...
    n_settlements = slice_limits[-1][1]
    warnings = 0

    np.random.seed(seed)
    for i in range(iterations):
        digits = np.random.choice(range(10), n_settlements)
        sim_likelihood = get_log_likelihood(digits, slice_limits,
                                            bottom_n, pe_seed, pe_iterations)
        sample.append(sim_likelihood)
        if np.isinf(sim_likelihood):
            print("Warning! Infinite simulated likelihood - perhaps increase the p.e. iterations!")
            warnings += 1
        if i % 50 == 0:
            print(i, np.mean(np.array(sample)[~np.isinf(sample)]))

    values, counts = np.unique(sample, return_counts=True)
    total = sum(counts)
    counts = np.cumsum(counts)

    def cdf(l):
        """ Forgiving CDF: P(L <= L_actual) """
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


if __name__ == "__main__":
    print(prob_of_entr(47, 2.1485))
    # expect ~ 0.133 (got with seed=1234) depending on the seed
