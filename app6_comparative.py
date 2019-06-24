from preprocessing import get_cleaned_data
from digit_entropy_distribution import get_entropy, prob_of_entr
from collections import OrderedDict
import pandas as pd


if not "df" in globals():
    df = get_cleaned_data()


def save_results(
        df_comparative,
        suspects,
        df_comparative_sorted_by_both,
        suspects2,
    ):
    df_comparative.to_csv("app6_comparative_result.csv", index=False)
    suspects.to_csv("app6_suspects.csv", index=False)
    df_comparative_sorted_by_both.to_csv(
        "app6_comparative_result_sort_by_both_incorr.csv",
        index=False
    )
    suspects2.to_csv("app6_suspects2.csv", index=False)


def load_results():
    """
    Returns four data frames.

    df_comparative: comparative data sorted by p_both ie. both
        years being regular, ascending ~ (1 - p_either)
    suspects: top suspects based on p_both
    df_comparative_sorted_by_both: comparative data sorted by
        p_both_incorr ie. both being incorrect, descending
    suspects2: top suspects based on p_both_incorr

    Other criteria: at least 8 electoral wards and at least 100
        Fidesz party voters in each ward in each suspect settlement
        in order to be considered,
    """
    df_comparative = pd.read_csv("app6_comparative_result.csv")
    suspects = pd.read_csv("app6_suspects.csv")
    df_comparative_sorted_by_both = pd.read_csv(
        "app6_comparative_result_sort_by_both_incorr.csv"
    )
    suspects2 = pd.read_csv("app6_suspects2.csv")
    return (
        df_comparative,
        suspects,
        df_comparative_sorted_by_both,
        suspects2
    )


def generate_data():
    agg_cols = OrderedDict([("ld_Fidesz", [get_entropy, len]),
                            ("Fidesz", min)])

    df_Fidesz_ent = df.groupby(["Telepules"]).aggregate(agg_cols)
    df_Fidesz_ent.reset_index(inplace=True)
    df_Fidesz_ent.columns = ["Settlement", "ld_entropy", "count", "min"]

    def calc_prob(row):
        return prob_of_entr(row["count"], row["ld_entropy"])

    """
    So two things cab be of interest:
    settlements with >= 50% of both being incorrect
    and those with   < 5% of both being correct
    """

    df_Fidesz_ent["count"] = df_Fidesz_ent["count"].astype(int)
    df_Fidesz_ent["prob_of_entr"] = df_Fidesz_ent.apply(calc_prob, axis=1)

    df_Fidesz_ent.sort_values(["prob_of_entr"], inplace=True)

    df_Fidesz_ent_2018 = pd.read_csv("Fidesz_entr_prob_2018.csv")

    df_comparative = pd.merge(df_Fidesz_ent, df_Fidesz_ent_2018,
                              how="inner", on=["Settlement"])
    df_comparative.columns = ["Settlement", "ld_Entropy_2019", "count_2019",
                              "min_votes_2019", "p_2019",
                              "ld_entropy_2018", "min_votes_2018",
                              "count_2018", "p_2018"]
    df_comparative["p_both"] = df_comparative.p_2019 * df_comparative.p_2018
    df_comparative["p_both_incorr"] = \
        (1 - df_comparative.p_2019) * (1 - df_comparative.p_2018)
    df_comparative.sort_values(["p_both"], inplace=True)

    # p_both: prob. of both being 'regular'
    # p_both_incorr: prob. of both being 'irregular'

    suspects = df_comparative.loc[(df_comparative.p_both < 0.1) &
                                  (df_comparative.min_votes_2019 >= 100 ) &
                                  (df_comparative.count_2018 >= 8)]
    df_comparative_sorted_by_both = \
        df_comparative.sort_values(["p_both_incorr"], ascending=[False])
    suspects2 = df_comparative_sorted_by_both.loc[
        (df_comparative_sorted_by_both.p_both_incorr >= 0.5) &
        (df_comparative_sorted_by_both.min_votes_2019 >= 100) &
        (df_comparative_sorted_by_both.count_2018 >= 8)
    ]
    save_results(df_comparative, suspects,
                 df_comparative_sorted_by_both, suspects2)


if __name__ == "__main__":
    generate_data()
