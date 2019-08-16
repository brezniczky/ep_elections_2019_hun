from collections import OrderedDict


def get_feasible_settlements(df, min_n_wards, min_fidesz_votes):
    d = OrderedDict()
    agg = (
        df[["Telepules", "Fidesz"]]
        .groupby(["Telepules"])
        .aggregate(OrderedDict([("Telepules", len), ("Fidesz", min)]))
    )
    agg.columns = ["n_wards", "min_fidesz_votes"]
    agg = agg.reset_index()
    return agg.loc[(agg.n_wards >= min_n_wards) &
                   (agg.min_fidesz_votes >= min_fidesz_votes)]["Telepules"]
