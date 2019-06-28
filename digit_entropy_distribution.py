from scipy.stats import entropy, poisson
import numpy as np
from functools import lru_cache


_DEFAULT_SEED = 1234
_DEFAULT_ITERATIONS = 10000


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


if __name__ == "__main__":
    print(prob_of_entr(47, 2.1485))
    # expect ~ 0.133 (got with seed=1234) depending on the seed
