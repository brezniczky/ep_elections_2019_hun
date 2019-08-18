import numpy as np
import pandas as pd
import numpy.random as rnd
from scipy import stats
from functools import lru_cache
from joblib import Memory


mem = Memory("./digit_correlations_cache", verbose=0)


@mem.cache()
def _get_digit_correlation_data(n_digits, seed=1234, n_iterations=10000):
    corrs = []
    rnd.seed(seed)
    for i in range(n_iterations):
        digits_1 = rnd.choice(range(10), n_digits)
        digits_2 = rnd.choice(range(10), n_digits)
        act_corr = np.corrcoef(digits_1, digits_2)
        corrs.append(act_corr[1, 0])

    corrs = sorted(corrs)
    return corrs


@lru_cache(1000)
def digit_correlation_cdf(n_digits, seed=1234, n_iterations=10000):
    # a small but efficient tribute to The Corrs :)
    corrs = _get_digit_correlation_data(n_digits, seed, n_iterations)

    def cdf(x):
        return np.digitize(x, corrs, right=False) / len(corrs)

    cdf.iterations = n_iterations
    cdf.max_x = corrs[-1]
    cdf.min_x = corrs[0]
    return cdf


def equality_rel_freq(a1: np.array, a2: np.array):
    ans = (a1 == a2).mean()
    return ans


@mem.cache()
def get_digit_equality_rel_freq_mc_data(n_digits, seed, n_iterations):
    raise Exception("This needs updating! "
                    "See digit_equality_prob_analytical_cdf ...")
    probs = []
    rnd.seed(seed)
    for i in range(n_iterations):
        digits_1 = rnd.choice(range(10), n_digits)
        digits_2 = rnd.choice(range(10), n_digits)
        act_rel_freq = (digits_1 == digits_2).mean()
        probs.append(act_rel_freq)

    probs = sorted(probs)
    return probs


# TODO: update the ipynb/html!
@lru_cache(1000)
def digit_equality_prob_mc_cdf(n_digits, seed=1234, n_iterations=50000):
    probs = get_digit_equality_rel_freq_mc_data(n_digits, seed, n_iterations)

    def cdf(x):
        return np.digitize(x, probs, right=False) / len(probs)

    cdf.iterations = n_iterations
    cdf.max_x = probs[-1]
    cdf.min_x = probs[0]
    return cdf


@lru_cache(1000)
def digit_equality_prob_analytical_cdf(n):
    """
    Probability of at least k digits being equal out of n pairs, described as
    the relative frequency k/n.

    :param n: number of pairs with i.i.d. p=0.1 equality events.
    :return: function telling the probability from the single relative frequency
        parameter.
    """
    inner_cdf = stats.binom(n, 0.1).cdf

    def cdf(rel_freq):
        # TODO: possibly do not operate on rel_freq... inaccurate values ~ waste
        return 1 - inner_cdf(int(round(rel_freq * n - 1)))

    return cdf


"""
faster, more accurate, no idea how to generalize for heterogeneous probabilities
"""
digit_equality_prob_cdf = digit_equality_prob_analytical_cdf


def correlation_prob_coeff_df(df: pd.DataFrame):
    ans_df = pd.DataFrame(columns=df.columns)
    ans_df["row_name"] = df.columns
    ans_df.set_index(["row_name"], inplace=True)

    cdf = digit_correlation_cdf(len(df))
    for row in df.columns:
        for col in df.columns:
            corr = np.corrcoef(df[row].values, df[col].values)[0, 1]
            try:
                ans_df.loc[row][col] = 1 - cdf(corr)
            except Exception as ex:
                import ipdb;
                ipdb.set_trace()
                print(ex)
    return ans_df


def equality_prob_coeff_df(df: pd.DataFrame):
    ans_df = pd.DataFrame(columns=df.columns)
    ans_df["row_name"] = df.columns
    ans_df.set_index(["row_name"], inplace=True)

    cdf = digit_equality_prob_cdf(len(df))
    for row in df.columns:
        for col in df.columns:
            prob = equality_rel_freq(df[row].values, df[col].values)
            try:
                ans_df.loc[row][col] = cdf(prob)
            except Exception as ex:
                import ipdb; ipdb.set_trace()
                print(ex)
                raise
    return ans_df


def equality_prob_vector(base_column: np.array, indep_columns: np.array):
    cdf = digit_equality_prob_cdf(len(base_column))
    ans = [
        cdf(equality_rel_freq(base_column, indep_column))
        for indep_column in indep_columns
    ]
    return np.array(ans)


def get_matrix_lambda_num(df: pd.DataFrame) -> float:
    """ Given a df with probabilities (outside its diagonals), calculate a
        "wavelength" ("lambda") value. It's like a 'period of recurrence' - for
        1:10 events, it would  be 10. Just adding the reciprocal of the
        probabilities in the matrix that df is, outside the diagonal. """
    return sum([1 / df[r][c] for r in df.index for c in df.columns if r != c])


def get_col_lambda_num(df, col_name) -> float:
    """ Similar to get_matrix_lambda_num, just a specified column.
        Division by zero problems are intentionally left unhandled.
    """
    s = df[df.index != col_name][col_name]
    return sum(1 / s)


def get_col_mean_prob(df, col_name) -> float:
    """ Return an average probability (geometric mean probability) generated
        from the contents of the column, except for the "diagonal" value. """
    s = np.exp(np.mean(np.log(
        df[df.index != col_name][col_name].values.astype(float)
    )))
    return s


def get_matrix_mean_prob(df: pd.DataFrame) -> float:
    ans = np.exp(np.mean([np.log(df[r][c])
                           for r in df.index
                           for c in df.columns if r != c]))
    return ans


if __name__ == "__main__":
    # cdf = digit_correlation_cdf(8531)
    # print("probability correlations higher than 0.01985", 1 - cdf(0.01985))
    # cdf = digit_equality_prob_mc_cdf(8531)
    # print("nonparam. probability equalities higher than 0.11", 1 - cdf(0.11))
    # cdf2 = digit_equality_prob_analytical_cdf(8531)
    # for i in range(100):
    #     x = cdf2(0.11)
    # print("param. probability equalities higher than 0.11", 1 - cdf2(0.11))

    rel_freq = equality_rel_freq(
        np.array([1, 2]),
        np.array([1, 2])
    )
    assert(rel_freq == 1.0)

    rel_freq = equality_rel_freq(
        np.array([1, 2]),
        np.array([1, 0])
    )
    assert(rel_freq == 0.5)

    cdf = digit_equality_prob_cdf(2)
    # P(rel_req=1, i.e. at least 2 matches out of 2) = 0.1 ^ 2
    assert(abs(cdf(1) - 0.01) < 0.0001)

    vec = equality_prob_vector(
        base_column=np.array([1, 2]),
        indep_columns=[np.array([1, 2]), np.array([1, 3]), np.array([10, 11])],
    )

    assert(sum(abs(vec - np.array([0.01, 0.19, 1.0]))) < 0.0001)
