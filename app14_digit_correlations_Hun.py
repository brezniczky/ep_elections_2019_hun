from preprocessing import get_preprocessed_data
from digit_correlations import (
    correlation_prob_coeff_df, equality_prob_coeff_df
)
import numpy as np


relevant_cols = ["Nevjegyzekben", "Ervenyes", "Fidesz", "Jobbik",
                 "LMP", "MSZP", "DK", "Momentum"]


def print_probs(df):
    pr_df = df.copy()
    pr_df.columns = [col[:8] for col in pr_df.columns]
    print(pr_df)


def check_correlations():
    df_full = get_preprocessed_data()
    np.random.seed(1234)
    df = df_full.loc[df_full["Fidesz"] +
                     np.random.choice(range(10), len(df_full)) - 5 > 100]
    print(len(df))
    df = df[["ld_" + col for col in relevant_cols]]
    corr_prob_df = correlation_prob_coeff_df(df)
    print(corr_prob_df)

    for col in corr_prob_df.columns:
        corr_prob_df[col][col] = 2

    l = list([list(x) for x in corr_prob_df.values])

    # zoom in on the DK-Jobbik correlation
    df = df_full.loc[df_full["DK"] +
                     np.random.choice(range(10), len(df_full)) - 5 > 50]
    df = df[["Jobbik", "DK"]]
    print(correlation_prob_coeff_df(df))
    # gives probability: zero

    # above a 100 votes on DK, you would naturally expect that to change
    df = df_full.loc[df_full["DK"] +
                     np.random.choice(range(10), len(df_full)) - 5 > 100]
    df = df[["Jobbik", "DK"]]
    print(correlation_prob_coeff_df(df))
    # still zero

    """
    OK, should return to checking corrs. as it may help, but of course there is
    a confounder - namely if one increases, the other too is likely, numbers of
    votes are generally correlated

    to be sure, equality could be checked
    """


def check_equalities():
    print("Fidesz >= 100")
    np.random.seed(1236)
    df = get_preprocessed_data()
    noise = np.random.choice(range(10), len(df)) - 5
    df = df.loc[df.Fidesz + noise >= 100]
    # df.ld_Fidesz = df.ld_LMP
    df = df[["ld_" + col for col in relevant_cols]]
    print(len(df))
    print_probs(equality_prob_coeff_df(df))

    print("LMP >= 20")
    df = get_preprocessed_data()
    noise = np.random.choice(range(10), len(df)) - 5
    df = df.loc[df.LMP + noise >= 20]
    # df.ld_Fidesz = df.ld_LMP
    df = df[["ld_" + col for col in relevant_cols]]
    print(len(df))
    print_probs(equality_prob_coeff_df(df))

    print("Ervenyes >= 200")
    df = get_preprocessed_data()
    noise = np.random.choice(range(10), len(df)) - 5
    df = df.loc[df.Ervenyes + noise >= 200]
    # df.ld_Fidesz = df.ld_LMP
    df = df[["ld_" + col for col in relevant_cols]]
    print(len(df))
    print_probs(equality_prob_coeff_df(df))

    print("100 <= Ervenyes < 200")
    df = get_preprocessed_data()
    noise = np.random.choice(range(10), len(df)) - 5
    df = df.loc[200 > (df.Ervenyes + noise)]
    noise = np.random.choice(range(10), len(df)) - 5
    df = df.loc[(df.Ervenyes + noise) >= 100]
    # df.ld_Fidesz = df.ld_LMP
    df = df[["ld_" + col for col in relevant_cols]]
    print(len(df))
    print_probs(equality_prob_coeff_df(df))

    print("Ervenyes >= 100")
    df = get_preprocessed_data()
    noise = np.random.choice(range(10), len(df)) - 5
    df = df.loc[df.Ervenyes + noise >= 100]
    # df.ld_Fidesz = df.ld_LMP
    df = df[["ld_" + col for col in relevant_cols]]
    print(len(df))
    print_probs(equality_prob_coeff_df(df))


if __name__ == "__main__":
    # check_correlations()
    check_equalities()
