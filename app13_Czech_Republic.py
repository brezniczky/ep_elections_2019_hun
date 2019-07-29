from CZ.scraper import load_or_generate_tabular_results
import digit_entropy_distribution as ded
import pandas as pd
from digit_entropy_distribution import LodigeTest
from digit_distribution_charts import plot_party_vote_by_digit_relationships
import numpy as np
"""

In [27]: len(set(tabular_results.district + tabular_results.municipality))
Out[27]: 1356

In [28]: len(set(tabular_results.municipality))
Out[28]: 1325

"""
def add_dm_key(df):
    df["dm_key"] = df.district + "/" + df.municipality


def load_results():
    res = pd.read_csv("CZ/party_results_tabular.csv")
    # construct a unique municipality key
    # as I TBH don't much like multi-level indexes ATM
    # maybe later?

    parties = [col
               for col in res.columns
               if col not in ["region", "district", "municipality", "ward"]]
    add_dm_key(res)
    return res, parties


def get_preprocessed_data():
    df, parties = load_results()
    for party in parties:
        df["ld_" + party] = df[party] % 10
    return df, parties


def rdig(along):
    return np.random.choice(10, len(along)) - 5


def get_suitable_municipalities(df, min_ward_count):
    municipality_ward_counts = (
        df
        .groupby(["district", "municipality"])
        .aggregate({"ward": len})
        .reset_index()
    )
    municipality_ward_counts = (
        municipality_ward_counts[
            municipality_ward_counts.ward > min_ward_count
        ]
    )
    return municipality_ward_counts[["district", "municipality"]]


def test_party(df, party):
    print("%s p-values" % party)
    test = LodigeTest(df[party], df["dm_key"], bottom_n=20, quiet=False)
    # overall p-value
    print("overall p-value", test.p)
    # p-value for

    iterations = 1000

    np.random.seed(1234)

    df_gt_100 = df[df[party] + rdig(df) >= 100]
    test = LodigeTest(df_gt_100[party], df_gt_100["dm_key"],
                      bottom_n=20, quiet=True, ll_iterations=iterations)
    print("p-value of votes >= 100", test.p)

    df_gt_50 = df[df[party] + rdig(df) >= 50]
    test = LodigeTest(df_gt_50[party], df_gt_50["dm_key"],
                      bottom_n=20, quiet=True, ll_iterations=iterations)
    print("p-value of votes >= 50", test.p)

    df_gt_75 = df[df[party] + rdig(df) >= 75]
    test = LodigeTest(df_gt_75[party], df_gt_75["dm_key"],
                      bottom_n=20, quiet=True, ll_iterations=iterations)
    print("p-value of votes >= 75", test.p)

    suitables = get_suitable_municipalities(df, 8)
    add_dm_key(suitables)

    df = df[df.dm_key.isin(suitables.dm_key)]
    print("for big enough municipalities (ward count >= 8)")

    df_gt_100 = df[df[party] + rdig(df) >= 100]
    test = LodigeTest(df_gt_100[party], df_gt_100["dm_key"],
                      bottom_n=20, quiet=True, ll_iterations=iterations)
    print("p-value of votes >= 100", test.p)

    df_gt_50 = df[df[party] + rdig(df) >= 50]
    test = LodigeTest(df_gt_50[party], df_gt_50["dm_key"],
                      bottom_n=20, quiet=True, ll_iterations=iterations)
    print("p-value of votes >= 50", test.p)

    df_gt_75 = df[df[party] + rdig(df) >= 75]
    test = LodigeTest(df_gt_75[party], df_gt_75["dm_key"],
                      bottom_n=20, quiet=True, ll_iterations=iterations)
    print("p-value of votes >= 75", test.p)


if __name__ == "__main__":
    df, parties = get_preprocessed_data()
    """ Spoiler alert: values don't even drop below 50%. """
    if input("run lengthy tests? (y/N)").lower().startswith("y"):
        test_party(df, "ANO 2011")
        test_party(df, "Obcanska demokraticka strana")
        test_party(df, "Ceska piratska strana")

    plot_party_vote_by_digit_relationships(df, "ANO 2011", n_bins=60)
    plot_party_vote_by_digit_relationships(df, "Obcanska demokraticka strana",
                                           n_bins=60)
    plot_party_vote_by_digit_relationships(df, "Ceska piratska strana",
                                           n_bins=60)
