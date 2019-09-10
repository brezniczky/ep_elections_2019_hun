from HU.preprocessing import get_preprocessed_data
from drdigit.digit_entropy_distribution import get_entropy, prob_of_entr
from collections import OrderedDict
import pandas as pd
from HU.AndrasKalman.load import load_2014
from arguments import load_output, save_output


if not "df" in globals():
    df = get_preprocessed_data()


def save_results(
        df_comparative,
        suspects,
        df_comparative_sorted_by_both,
        suspects2,
        df_comparative_sorted_by_all_3,
        suspects3
    ):
    save_output(df_comparative, "app6_comparative_result.csv")
    save_output(suspects, "app6_suspects.csv")
    save_output(
        df_comparative_sorted_by_both,
        "app6_comparative_result_sort_by_both_incorr.csv"
    )
    save_output(suspects2, "app6_suspects2.csv")
    save_output(df_comparative_sorted_by_all_3,
                "app6_comparative_sorted_by_all_3.csv")
    save_output(suspects3, "app6_suspects3.csv")


def load_results():
    """
    Returns four data frames.

    df_comparative: comparative data sorted by p_both ie. both
        years being regular, ascending ~ (1 - p_either)
    suspects: top suspects based on p_both
    df_comparative_sorted_by_both: comparative data sorted by
        p_both_incorr ie. both being incorrect, descending
    suspects2: top suspects based on p_both_incorr
    df_comparative_sorted_by_all_3: comparative data sorted by
        p_all_3 i.e. all 3 (2014, 2018, 2019) being regular
    suspects3: top candidates from df_comparative_sorted_by_all_3.

    Other criteria: at least 8 electoral wards and at least 100 Fidesz
        party votes (in 2019) in each ward in each suspect settlement
        in order to be considered,
    """

    df_comparative = load_output("app6_comparative_result.csv")
    suspects = load_output("app6_suspects.csv")
    df_comparative_sorted_by_both = load_output(
        "app6_comparative_result_sort_by_both_incorr.csv"
    )
    suspects2 = load_output("app6_suspects2.csv")

    df_comparative_sorted_by_all_3 = load_output(
        "app6_comparative_sorted_by_all_3.csv"
    )
    suspects3 = load_output("app6_suspects3.csv")

    return (
        df_comparative,
        suspects,
        df_comparative_sorted_by_both,
        suspects2,
        df_comparative_sorted_by_all_3,
        suspects3,
    )


def calc_prob(row):
    return prob_of_entr(row["count"], row["ld_entropy"])


def load_2014_data():
    agg_cols = OrderedDict([("ld_Fidesz", [get_entropy, len]),
                            ("Fidesz", min)])

    df_2014 = load_2014()
    df_2014["Fidesz"] = df_2014["Fidesz"].astype(int)
    df_2014["ld_Fidesz"] = df_2014["Fidesz"] % 10
    df_Fidesz_ent_2014 = df_2014.groupby(["Telepules"]).aggregate(agg_cols)
    df_Fidesz_ent_2014.reset_index(inplace=True)
    df_Fidesz_ent_2014.columns = ["Settlement", "ld_entropy", "count", "min"]
    df_Fidesz_ent_2014["count"] = df_Fidesz_ent_2014["count"].astype(int)
    df_Fidesz_ent_2014["prob_of_entr"] = df_Fidesz_ent_2014.apply(calc_prob, axis=1)

    return df_Fidesz_ent_2014


def generate_data():
    agg_cols = OrderedDict([("ld_Fidesz", [get_entropy, len]),
                            ("Fidesz", min)])

    df_Fidesz_ent = df.groupby(["Telepules"]).aggregate(agg_cols)
    df_Fidesz_ent.reset_index(inplace=True)
    df_Fidesz_ent.columns = ["Settlement", "ld_entropy", "count", "min"]

    """
    So two things cab be of interest:
    settlements with >= 50% of both being incorrect
    and those with   < 5% of both being correct
    """

    df_Fidesz_ent["count"] = df_Fidesz_ent["count"].astype(int)
    df_Fidesz_ent["prob_of_entr"] = df_Fidesz_ent.apply(calc_prob, axis=1)

    df_Fidesz_ent.sort_values(["prob_of_entr"], inplace=True)

    df_Fidesz_ent_2018 = load_output("Fidesz_entr_prob_2018.csv")
    df_Fidesz_ent_2014 = load_2014_data()


    df_comparative = pd.merge(df_Fidesz_ent, df_Fidesz_ent_2018,
                              how="inner", on=["Settlement"])
    df_Fidesz_ent_2014.columns = [
        column + "_z" if column != "Settlement" else column
        for column in df_Fidesz_ent_2014.columns
    ]
    df_comparative = pd.merge(df_comparative, df_Fidesz_ent_2014,
                              how="inner", on=["Settlement"], suffixes=["", "_z"])

    df_comparative.columns = ["Settlement", "ld_Entropy_2019", "count_2019",
                              "min_votes_2019", "p_2019",
                              "ld_entropy_2018", "min_votes_2018",
                              "count_2018", "p_2018",
                              "ld_Entropy_2014", "count_2014",
                              "min_votes_2014", "p_2014"]

    df_comparative["p_all_3"] = \
        df_comparative.p_2019 * df_comparative.p_2018 * df_comparative.p_2014
    df_comparative["p_all_2"] = df_comparative.p_2019 * df_comparative.p_2018
    df_comparative["p_both_incorr"] = \
        (1 - df_comparative.p_2019) * (1 - df_comparative.p_2018)
    df_comparative.sort_values(["p_all_2"], inplace=True)

    # p_both: prob. of both being 'regular'
    # p_both_incorr: prob. of both being 'irregular'

    suspects = df_comparative.loc[(df_comparative.p_all_2 < 0.1) &
                                  (df_comparative.min_votes_2019 >= 100 ) &
                                  (df_comparative.count_2018 >= 8)]
    df_comparative_sorted_by_both = \
        df_comparative.sort_values(["p_both_incorr"], ascending=[False])
    suspects2 = df_comparative_sorted_by_both.loc[
        (df_comparative_sorted_by_both.p_both_incorr >= 0.5) &
        (df_comparative_sorted_by_both.min_votes_2019 >= 100) &
        (df_comparative_sorted_by_both.count_2018 >= 8)
    ]

    df_comparative_sorted_by_all_3 = df_comparative.sort_values(["p_all_3"])
    suspects3 = df_comparative_sorted_by_all_3.loc[
        (df_comparative_sorted_by_all_3.p_all_3 <= 0.1) &
        (df_comparative_sorted_by_both.min_votes_2019 >= 100) &
        (df_comparative_sorted_by_both.count_2018 >= 8)
    ]

    save_results(df_comparative, suspects,
                 df_comparative_sorted_by_both, suspects2,
                 df_comparative_sorted_by_all_3, suspects3)


if __name__ == "__main__":
    generate_data()
