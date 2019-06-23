from scipy.stats import entropy
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
