import pandas as pd
from preprocessing import get_preprocessed_data
from AndrasKalman.load import load_2014
from explore_2018 import load_2018_fidesz
from digit_entropy_distribution import prob_of_twins
from collections import OrderedDict


def get_twins(df):
    agg_cols = OrderedDict([
        ("ld_Fidesz", [prob_of_twins, len]),
        ("Fidesz", min)
    ])

    df = \
        df[["Telepules", "ld_Fidesz", "Fidesz"]] \
        .groupby("Telepules") \
        .aggregate(agg_cols) \
        .reset_index()
    df.columns = \
        ["Telepules", "p_ld_Fidesz_twins", "n_wards", "min_Fidesz_votes"]
    df.sort_values(["p_ld_Fidesz_twins"], inplace=True)

    return df


def get_suspects(df, p=0.05):
    return df.loc[(df.min_Fidesz_votes_2019 >= 100) &
                  (df.n_wards_2019 >= 8) &
                  (df.p_all_regular <= p)]


# def save_results(
#         df_2014_p_twins, df_2018_p_twins, df_2019_p_twins,
#         suspects_2014, suspects_2018, suspects_2019):
#     df_2014_p_twins.to_csv("app8_2014_twins.csv", index=False)
#     df_2018_p_twins.to_csv("app8_2018_twins.csv", index=False)
#     df_2019_p_twins.to_csv("app8_2019_twins.csv", index=False)
#     suspects_2014.to_csv("app8_suspects_2014.csv", index=False)
#     suspects_2018.to_csv("app8_suspects_2018.csv", index=False)
#     suspects_2019.to_csv("app8_suspects_2019.csv", index=False)
def save_results(df_p_twins, suspects):
    df_p_twins.to_csv("app8_twins.csv", index=False)
    suspects.to_csv("app8_suspects.csv", index=False)


# def load_results():
#     df_2014_p_twins = pd.read_csv("app8_2014_twins.csv")
#     df_2018_p_twins = pd.read_csv("app8_2018_twins.csv")
#     df_2019_p_twins = pd.read_csv("app8_2019_twins.csv")
#     suspects_2014 = pd.read_csv("app8_suspects_2014.csv")
#     suspects_2018 = pd.read_csv("app8_suspects_2018.csv")
#     suspects_2019 = pd.read_csv("app8_suspects_2019.csv")
#     return (
#         df_2014_p_twins, df_2018_p_twins, df_2019_p_twins,
#         suspects_2014, suspects_2018, suspects_2019,
#     )
def load_results():
    df_p_twins = pd.read_csv("app8_twins.csv")
    suspects = pd.read_csv("app8_suspects.csv")
    return df_p_twins, suspects


if __name__ == "__main__":
    df_2014 = load_2014()
    df_2014["ld_Fidesz"] = df_2014["Fidesz"] % 10
    df_2018 = load_2018_fidesz()
    df_2018.rename(columns={"Settlement": "Telepules",
                            "ld": "ld_Fidesz",
                            "Votes": "Fidesz"},
                   inplace=True)
    df_2019 = get_preprocessed_data()

    df_2014_p_twins = get_twins(df_2014)
    df_2018_p_twins = get_twins(df_2018)
    df_2019_p_twins = get_twins(df_2019)

    # suspects_2014 = get_suspects(df_2014_p_twins, 0.1)
    # suspects_2018 = get_suspects(df_2018_p_twins, 0.1)
    # suspects_2019 = get_suspects(df_2019_p_twins, 0.1)

    # save_results(
    #     df_2014_p_twins, df_2018_p_twins, df_2019_p_twins,
    #     suspects_2014, suspects_2018, suspects_2019
    # )

    df_2019_p_twins_cols_with_suffix = {
        column: column + "_2019"
        for column in df_2019_p_twins.columns
        if column != "Telepules"
    }

    df_comparative = pd.merge(df_2014_p_twins, df_2018_p_twins,
                              on="Telepules", how="inner",
                              suffixes=["_2014", "_2018"])
    df_comparative = pd.merge(df_comparative,
                              df_2019_p_twins.rename(
                                    columns=df_2019_p_twins_cols_with_suffix
                              ),
                              on="Telepules", how="inner")
    df_comparative["p_all_regular"] = (
        df_comparative.p_ld_Fidesz_twins_2014 *
        df_comparative.p_ld_Fidesz_twins_2018 *
        df_comparative.p_ld_Fidesz_twins_2019
    )

    df_comparative.sort_values(["p_all_regular"], inplace=True)
    suspects = get_suspects(df_comparative)
    save_results(df_comparative, suspects)
