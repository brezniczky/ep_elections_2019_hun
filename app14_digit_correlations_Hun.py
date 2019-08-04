from preprocessing import get_preprocessed_data
from digit_correlations import (
    correlation_prob_coeff_df, equality_prob_coeff_df,
    get_matrix_lambda_num, get_col_lambda_num, get_col_mean_prob,
    get_matrix_mean_prob
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


relevant_cols = ["Nevjegyzekben", "Ervenyes", "Fidesz", "Jobbik",
                 "LMP", "MSZP", "DK", "Momentum"]

ACT_FILENAME = "app14_valid_votes_coincidences.csv"
BASELINE_FILENAME_FORMAT = ("app14_simulated_baseline/" +
                            "app14_valid_votes_coincidences_baseline_%d.csv")
N_BASELINE_REPEATS = 200


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


def test_cross_equalities(df):
    df = (
        equality_prob_coeff_df(
            )
    )
    df.rename(columns={
        "ld_Nevjegyzekben": "ld_Nevj", "ld_Ervenyes": "ld_Erv",
        "ld_Momentum": "ld_Mom", "ld_Mi Hazank": "ld_Mi Haz"},
        inplace = True
    )
    return df


def test_area(df, area, starts_with_only=False):
    df_act = df[
        df.Telepules.str.startswith(area)
        if starts_with_only else
        df[df.Telepules == area]
    ]
    df = (
        equality_prob_coeff_df(
        df_act[["ld_Fidesz", "ld_Jobbik", "ld_Mi Hazank", "ld_LMP", "ld_DK",
                "ld_Ervenyes", "ld_Nevjegyzekben", "ld_MSZP", "ld_Momentum"]])
    )
    df.rename(columns={
        "ld_Nevjegyzekben": "ld_Nevj", "ld_Ervenyes": "ld_Erv",
        "ld_Momentum": "ld_Mom", "ld_Mi Hazank": "ld_Mi Haz"},
        inplace = True
    )
    return df


def check_equalities_2(settlement_filter = None, df = None):
    if df is None:
        df = get_preprocessed_data()
    if settlement_filter is not None:
        df = df[df.Telepules.isin(settlement_filter)]
    np.random.seed(1235)
    noise = np.random.choice(range(10), len(df)) - 5
    df = df[df.Ervenyes + noise >= 100]
    ans_rows = []
    for settlement in set(df.Telepules):
        df_act = df[df.Telepules == settlement]
        act_min_Ervenyes = min(df_act.Ervenyes)

        if len(df_act) < 8:
            continue
        if act_min_Ervenyes < 100:
            continue

        df_act = df_act[
            ["ld_Nevjegyzekben", "ld_Ervenyes",
             "ld_Fidesz", "ld_DK", "ld_Momentum", "ld_LMP", "ld_Jobbik",
             "ld_MSZP"]
        ]
        test_res = equality_prob_coeff_df(df_act)
        lam = get_matrix_lambda_num(test_res)
        erv_lam = get_col_lambda_num(test_res, "ld_Ervenyes")
        erv_mean_prob = get_col_mean_prob(test_res, "ld_Ervenyes")
        fidesz_mean_prob = get_col_mean_prob(test_res, "ld_Fidesz")
        mean_prob = get_matrix_mean_prob(test_res)
        row = test_res["ld_Ervenyes"]
        row.index = ["ld_Erv_" + str(k) for k in row.index]
        row2 = test_res["ld_Fidesz"]
        row2.index = ["ld_Fid_" + str(k) for k in row2.index]
        row = row.to_dict()
        row.update(row2.to_dict())
        row["lambda"] = lam
        row["Erv_lambda"] = erv_lam
        row["Erv_mean_prob"] = erv_mean_prob
        row["Fidesz_mean_prob"] = fidesz_mean_prob
        row["mean_prob"] = mean_prob
        row["Telepules"] = settlement
        row["n_wards"] = len(df_act)
        row["min_ervenyes"] = act_min_Ervenyes
        ans_rows.append(row)
        print(row)

    return pd.DataFrame(ans_rows)


def get_baseline_filename(i):
    return BASELINE_FILENAME_FORMAT % i


def get_act_data():
    if not os.path.exists(ACT_FILENAME):
        df_act = check_equalities_2()
        df_act.to_csv(ACT_FILENAME, index=False)
    else:
        df_act = pd.read_csv(ACT_FILENAME)
    return df_act


def require_simulations():
    for i in range(N_BASELINE_REPEATS):
        filename = get_baseline_filename(i)
        if os.path.exists(filename):
            continue

        df = get_preprocessed_data()

        def random_digits_df():
            nonlocal df
            return np.random.choice(range(10), len(df))

        np.random.seed(1234 + i)
        df.ld_Momentum = random_digits_df()
        df.ld_DK = random_digits_df()
        df.ld_LMP = random_digits_df()
        df.ld_Jobbik = random_digits_df()
        df.ld_Momentum = random_digits_df()
        df.ld_MSZP = random_digits_df()
        df.ld_Ervenyes = random_digits_df()

        df = check_equalities_2(df=df)
        df.to_csv(filename,
                  index=False)


def plot_equality_tests():

    def get_mean_mean_prob(test_results):
        p1 = test_results.Erv_mean_prob
        p2 = test_results.Fidesz_mean_prob
        # probably this should be better as a geom. mean too but ... approx.
        # equal I guess due to the value range (from experience) anyway and I'm
        # lazy
        return np.mean(p1 * p2 ** 0.5)

    mean_mean_probs = []
    require_simulations()
    for i in range(N_BASELINE_REPEATS):
        df = pd.read_csv(get_baseline_filename(i))
        mean_mean_prob = get_mean_mean_prob(df)
        mean_mean_probs.append(mean_mean_prob)

    df_act = get_act_data()
    mean_mean_prob = get_mean_mean_prob(df_act)

    hh = plt.hist(mean_mean_probs, bins=30)
    plt.title("Mean probability distribution: simulated vs. actual")
    plt.axvline(x=mean_mean_prob, color="red")
    plt.text(mean_mean_prob + 0.0005, max(hh[0]) * 0.95,
             "P=%.2f %%" %
             (100 * sum(mean_mean_probs <= mean_mean_prob) /
              len(mean_mean_probs)))
    plt.show()


def get_baseline_probs():

    for i in range(N_BASELINE_REPEATS):
        df = pd.read_csv(get_baseline_filename(i))
        # df.set_index(["Telepules"])
        if i == 0:
            if sum(df["Fidesz_mean_prob"].values == 0) > 0:
                import ipdb; ipdb.set_trace()
            if sum(df["Fidesz_mean_prob"].values == 0) > 0:
                import ipdb;
            Fid_probs = np.log(df["Fidesz_mean_prob"].values)
            Erv_probs = np.log(df["Erv_mean_prob"].values)
        else:
            if sum(df["Fidesz_mean_prob"].values == 0) > 0:
                import ipdb; ipdb.set_trace()
            if sum(df["Fidesz_mean_prob"].values == 0) > 0:
                import ipdb;
            Fid_probs += np.log(df["Fidesz_mean_prob"].values)
            Erv_probs += np.log(df["Erv_mean_prob"].values)

    return np.exp((Fid_probs + Erv_probs) / N_BASELINE_REPEATS / 2)


def get_top_list(n_top=None):
    df_act = get_act_data()
    baseline_probs = get_baseline_probs()

    top_list = df_act[["Telepules", "mean_prob"]].copy()

    top_list["rel_mean_prob"] = top_list["mean_prob"] / baseline_probs

    top_list = top_list.sort_values(["rel_mean_prob"])
    if n_top is not None:
        top_list = top_list.head(n_top)
    top_list.reset_index()
    top_list.index = range(1, len(top_list) + 1)
    return df_act, top_list


def get_digit_freqs():
    df = get_preprocessed_data()
    np.random.seed(4321)
    df = df[df.Fidesz >= 100 + np.random.choice(range(10), len(df)) - 5]

    suited = df[["Telepules", "Fidesz"]].groupby(["Telepules"]).aggregate(
        len).reset_index()
    suited = suited[suited.Fidesz >= 8].Telepules

    df = df[df.Telepules.isin(suited.values)]

    ld_Fidesz_freqs = np.unique(df.ld_Fidesz, return_counts=True)[1]
    ld_Fidesz_freqs = ld_Fidesz_freqs / sum(ld_Fidesz_freqs)
    ld_Erv_freqs = np.unique(df.ld_Ervenyes, return_counts=True)[1]
    ld_Erv_freqs = ld_Erv_freqs / sum(ld_Erv_freqs)

    return ld_Fidesz_freqs, ld_Erv_freqs


if __name__ == "__main__":
    # check_correlations()
    # check_equalities()

    # df = check_equalities_2(
    #     settlement_filter=["Eger", "Tata", "Sopron", "Gyöngyös", "Keszthely"]
    # )
    # plot_equality_tests()
    # df, df_top_list = get_top_list()
    print(get_top_list())
