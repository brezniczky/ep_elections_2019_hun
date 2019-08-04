import numpy as np
import pandas as pd
import numpy.random as rnd
from scipy import stats
from functools import lru_cache


@lru_cache(100)
def digit_correlation_cdf(n_digits, seed=1234, n_iterations=10000):
    corrs = []
    rnd.seed(seed)
    for i in range(n_iterations):
        digits_1 = rnd.choice(range(10), n_digits)
        digits_2 = rnd.choice(range(10), n_digits)
        act_corr = np.corrcoef(digits_1, digits_2)
        corrs.append(act_corr[1, 0])

    corrs = sorted(corrs)

    def cdf(x):
        return np.digitize(x, corrs, right=False) / len(corrs)

    cdf.iterations = n_iterations
    cdf.max_x = corrs[-1]
    cdf.min_x = corrs[0]
    return cdf


def equality_prob(a1: np.array, a2: np.array):
    return (a1 == a2).mean()


@lru_cache(100)
def digit_equality_prob_mc_cdf(n_digits, seed=1234, n_iterations=50000):
    corrs = []
    rnd.seed(seed)
    for i in range(n_iterations):
        digits_1 = rnd.choice(range(10), n_digits)
        digits_2 = rnd.choice(range(10), n_digits)
        act_prob = (digits_1 == digits_2).mean()
        corrs.append(act_prob)

    corrs = sorted(corrs)

    def cdf(x):
        return np.digitize(x, corrs, right=False) / len(corrs)

    cdf.iterations = n_iterations
    cdf.max_x = corrs[-1]
    cdf.min_x = corrs[0]
    return cdf


def digit_equality_prob_analytical_cdf(n):
    inner_cdf = stats.binom(n, 0.1).cdf

    def cdf(rel_freq):
        return inner_cdf(rel_freq * n)

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
            prob = equality_prob(df[row].values, df[col].values)
            try:
                ans_df.loc[row][col] = 1 - cdf(prob)
            except Exception as ex:
                import ipdb;
                ipdb.set_trace()
                print(ex)
    return ans_df


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
    cdf = digit_correlation_cdf(8531)
    print("probability correlations higher than 0.01985", 1 - cdf(0.01985))
    cdf = digit_equality_prob_mc_cdf(8531)
    print("nonparam. probability equalities higher than 0.11", 1 - cdf(0.11))
    cdf2 = digit_equality_prob_analytical_cdf(8531)
    for i in range(100):
        x = cdf2(0.11)
    print("param. probability equalities higher than 0.11", 1 - cdf2(0.11))
