import numpy as np
import pandas as pd
import numpy.random as rnd
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
def digit_equality_prob_cdf(n_digits, seed=1234, n_iterations=50000):
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


if __name__ == "__main__":
    cdf = digit_correlation_cdf(8531)
    print("probability correlations higher than 0.01985", 1 - cdf(0.01985))
    cdf = digit_equality_prob_cdf(8531)
    print("probability equalities higher than 0.1", 1 - cdf(0.1))
