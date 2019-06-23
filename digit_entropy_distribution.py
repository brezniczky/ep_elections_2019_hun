from scipy.stats import entropy
import numpy as np
from functools import lru_cache


_DEFAULT_SEED = 1234
_DEFAULT_ITERATIONS = 10000


def get_entropy(x):
    """ x should be a list-like of digits """
    digits, counts = np.unique(x, return_counts=True)
    counts = counts / sum(counts)
    # no need to pad for missing digits, terms with a zero coeff.
    # fall out anyway
    #
    # if len(counts) < 10:
    #     counts = np.concatenate([counts, [0] * (10 - len(counts))], axis=0)
    return entropy(counts)


@lru_cache(maxsize=1000)
def get_cdf(n_wards,
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


def prob_of_entr(n_wards, entr,
                 seed=_DEFAULT_SEED,
                 iterations=_DEFAULT_ITERATIONS):
    cdf = get_cdf(n_wards, seed, iterations)
    return sum(cdf <= entr) / len(cdf)


if __name__ == "__main__":
    print(prob_of_entr(47, 2.1485))
