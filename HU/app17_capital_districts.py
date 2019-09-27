import numpy as np
from HU.app15_overall_ranking import get_overall_list_last_2_years
from HU.cleaning import get_2019_cleaned_data, get_2018_cleaned_data
from arguments import save_output
from matplotlib import pyplot as plt


def _get_budapest_ranking():
    df = get_overall_list_last_2_years()
    df = df[["Telepules", "p"]]
    df = df[df.Telepules.str.startswith("Budapest")].reset_index().drop(columns="index")
    return df


def _get_district_stats(df, election_df):
    # inner join df with the ep_data,
    # add column: turnout
    # aggregate by district, for the following measures:
    #
    # total valid votes   (A)
    # total registered voters  (B)
    # mean turnout
    # fidesz mean votes     # helps determining if the range affected the digit distr.
                            # low range ~> possibly high unevenness
    # fidesz median votes
    # fidesz vote share
    # sd(fidesz vote share)
    # ward count
    #
    # add column: overall turnout (A/B)

    return (
        df.merge(election_df, how="left", on="Telepules")
          .assign(Turnout=lambda x: x.Ervenyes / x.Nevjegyzekben)
          .assign(Vote_share=lambda x: x.Fidesz / x.Ervenyes)
    )


def _get_election_df(year):
    if year == 2018:
        election_df = get_2018_cleaned_data()
        election_df.rename(
            columns={
                "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT": "Fidesz"
            }, inplace=True
        )
    elif year == 2019:
        election_df = get_2019_cleaned_data()

    return election_df


def get_ward_stats(year=2019, top_district_indexes=None):
    districts_df = _get_budapest_ranking()
    if top_district_indexes is not None:
        districts_df = districts_df.iloc[top_district_indexes]
    election_df = _get_election_df(year)
    ans = _get_district_stats(districts_df, election_df)
    ans = districts_df.merge(ans, how="left", on="Telepules")
    return ans


def get_district_stats(year=2019):
    districts_df = _get_budapest_ranking()
    election_df = _get_election_df(year)

    ans = _get_district_stats(districts_df, election_df)
    ans = (
        ans.groupby(["Telepules"])
           .agg({"Fidesz": [sum, "mean", "median", min, max],
                 "Ervenyes": [sum],
                 "Nevjegyzekben": [sum, len],  # len gives the ward count
                 "Turnout": ["mean", "median"],
                 "Vote_share": [np.std],  # stdev of per ward vote shares
                 })
           .assign(Vote_share=lambda x: x[("Fidesz", "sum")] /
                                        x[("Ervenyes", "sum")])
    )

    # a benefit; restores the original sort order
    ans = districts_df.merge(ans, how="left", on="Telepules")

    return ans


def plot_vote_share_histograms(year=2019, weighted=False):
    df = _get_budapest_ranking()
    election_df = _get_election_df(year)
    stats = _get_district_stats(df, election_df)

    nrows = int(np.ceil(len(df) / 4))

    fig, axes = plt.subplots(nrows=nrows, ncols=4, sharex=True, sharey=True)

    axes = np.concatenate(axes)

    # bins = np.linspace(0, election_df["Fidesz"].max(), num=40)
    bins = np.linspace(0, 1, num=40)
    try:
        for district, axis in zip(df.Telepules, axes[:len(df)]):
            axis.set_title(district[len("Budapest "):])
            act_stats = stats[stats.Telepules==district]
            weights = None
            if weighted:
                weights = act_stats["Fidesz"]
                print("weighted")
            axis.hist(act_stats["Fidesz"] / act_stats["Ervenyes"],
                      bins=bins, density=True, weights=weights)
            # quantile plot?

        fig.tight_layout()
        plt.show()
    finally:
        plt.close()


def plot_aggregated_histogram_halves(
    year=2019, weighted=False, turnout=False, print_info=False,
    party="Fidesz"
):

    df = _get_budapest_ranking()
    election_df = _get_election_df(year)
    stats = _get_district_stats(df, election_df)

    fig, axis = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

    n_top = 15
    # facs = (1, 23 / (23 - 16) * 0.7)
    # facs = (1, 2.1)  # 2019 unweighted
    # facs = (1, 2.7)  # 2019 weighted
    facs = (1, 1)
    top_cities = df.Telepules[:n_top]
    bottom_cities = df.Telepules[n_top:]
    cols = [(0, 0, 1, 0.5), (1, 1, 0, 0.5)]

    bins = np.linspace(0, 1, num=40)
    try:
        by_what_str = "turnout" if turnout else ("vote share of party")
        if weighted:
            title = ("%d distribution of party votes by %s" %
                     (year, by_what_str))
        else:
            title = ("%d distribution of electoral wards by %s" %
                     (year, by_what_str))
        plt.title(title + "\n" + party)

        for districts, col, fac in zip([top_cities, bottom_cities], cols, facs):
            # axis.set_title(district[len("Budapest "):])
            act_stats = stats[stats.Telepules.isin(districts)]
            if print_info:
                print("sum party votes:", sum(act_stats[party]))
                print("nr of wards:", len(act_stats))
            weights = [fac] * len(act_stats)
            if weighted:
                weights = act_stats[party] * fac
                print("weighted")
            if not turnout:
                values = act_stats[party] / act_stats["Ervenyes"]
            else:
                values = act_stats["Ervenyes"] / act_stats["Nevjegyzekben"]
            axis.hist(values,
                      bins=bins, weights=weights,
                      color=col)
        axis.legend(["top", "bottom"])
        plt.xlabel(by_what_str)
        if weighted:
            plt.ylabel("nr of votes")
        else:
            plt.ylabel("nr of wards")

        fig.tight_layout()
        plt.show()
    finally:
        plt.close()


def plot_digits_and_dist(votes, title):

    plt.rcParams["figure.figsize"] = [7, 3]
    vote_bins = np.linspace(0, 400, 40)

    fig, axes = plt.subplots(nrows=1, ncols=2)

    ax_l, ax_r = axes

    ax_l.hist(votes % 10, bins=range(11))
    ax_l.set_xlabel("Last digit")
    ax_l.set_ylabel("Frequency")

    ax_l.set_xticks(np.arange(0, 10) + 0.5)
    ax_l.set_xticklabels(np.arange(0, 10))

    ax_r.hist(votes, bins=vote_bins)
    ax_r.set_xlabel("Votes")
    ax_r.set_ylabel("Frequency")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # district_stats_2018_df = get_district_stats(2018)
    # save_output(district_stats_2018_df, "app17_2018_district_stats.csv")
    # district_stats_2019_df = get_district_stats(2019)
    # save_output(district_stats_2019_df, "app17_2019_district_stats.csv")

    # # with or without weighting, "by the eye" - a "unimodal" top at least
    # # one bar away to the left from 0.4 is "below 0.4"
    # #
    # # 3 "significantly" below 0.4 (top half) vs.
    # # 6 "significantly" below 0.4 (bottom half)
    # plot_vote_share_histograms(year=2018, weighted=True)

    # with or without weighting, "by the eye" - a "unimodal" top at least
    # one bar away to the left from 0.4 is "below 0.4"
    #
    # 3 "significantly" below 0.4 (top half) vs.
    # 6 "significantly" below 0.4 (bottom half)
    # plot_vote_share_histograms(year=2019, weighted=True)

    # plot_aggregated_histogram_halves(2018, weighted=False)
    # plot_aggregated_histogram_halves(2018, weighted=True, turnout=False)

    plot_aggregated_histogram_halves(2018, weighted=True, turnout=False, party="MOMENTUM MOZGALOM")

    plot_aggregated_histogram_halves(2018, weighted=True, turnout=True, party="MOMENTUM MOZGALOM")
