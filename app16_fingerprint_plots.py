from preprocessing import get_preprocessed_data
from matplotlib.colors import Normalize
from cleaning import get_2014_cleaned_data, get_2018_cleaned_data
import matplotlib.pyplot as plt
import numpy as np
import os
import app15_overall_ranking as app15

# todo: 1. an insane amount of refactoring :) :(


_ranking = app15.get_overall_list_last_2_years()


FINGERPRINT_DIR = "fingerprints"


PARTIES_2014 = ["Fidesz-KDNP", "Jobbik", "LMP", "Együtt 2014",
                'MSZP-Együtt-DK-PM-MLP']
# PARTIES_2014 = []

PARTIES_2018 = [
    "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT",
    'JOBBIK MAGYARORSZÁGÉRT MOZGALOM',
    'LEHET MÁS A POLITIKA',
    'MOMENTUM MOZGALOM',
    'DEMOKRATIKUS KOALÍCIÓ',
    'EGYÜTT - A KORSZAKVÁLTÓK PÁRTJA',
    "MAGYAR SZOCIALISTA PÁRT-PÁRBESZÉD MAGYARORSZÁGÉRT PÁRT"
]

PARTIES_2019 = ["Fidesz", "Jobbik", "LMP", "Momentum", "DK",
                "MSZP", "Mi Hazank"]
# PARTIES_2019 = []


ZOOM_ONTO = [
    "Együtt 2014", 'EGYÜTT - A KORSZAKVÁLTÓK PÁRTJA'
]


def plot_fingerprint(winner_votes, valid_votes,
                     registered_voters, title, filename,
                     weighted=True,
                     zoom_onto=False):

    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]
    if zoom_onto:
        bins[1] = 0.4 * bins[1]

    weights = None if not weighted else winner_votes
    plt.hist2d(
        # winner_votes / registered_voters,  # TODO: or valid_votes?
        valid_votes / registered_voters,
        winner_votes / valid_votes,  # TODO: or valid_votes?
        bins=bins,
        weights=weights
    )
    plt.title(title)
    plt.savefig(os.path.join(FINGERPRINT_DIR, filename))
    plt.show()


def get_diffs(x1, y1, w1, x2, y2, w2):
    # TODO: zooming

    # 1. histogram gets x arr, y arr, weights
    #    returns values (nx x ny), x bin edges (n+1), y bin edges (n+1)
    #    will spec the bins on input so that they conform on return for 1 and 2
    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]

    v1, x1, y1 = np.histogram2d(x1, y1, bins=bins, weights=w1)
    v2, x2, y2 = np.histogram2d(x2, y2, bins=bins, weights=w2)

    assert all(x1 == x2) and all(y1 == y2)

    # 2. calc. intersection (inert area)
    def is_(x):
        # median or mean as a fallback to act as a threshold of
        # 'surely sufficiently convincingly included'
        mx = np.median(x)
        if mx == 0:
            mx = np.mean(x)
        # 0..1 value for AND-style multiplicability
        return np.minimum(x / mx, 1)

    # operate on slightly fuzzy logical masks here
    is_intersection = is_(v1) * is_(v2)
    is_v1_only = is_(v1) * (1 - is_intersection)
    is_v2_only = is_(v2) * (1 - is_intersection)

    # the product is still in 0..1 and can be used to slice the A or B-specific
    return v1 * is_v1_only, v2 * is_v2_only, x1, y1


def plot_histogram2d(d_2_1, d_1_2, binx, biny, show=True, filename=None):
    # plot a numpy histogram2d result via matplotlib
    # TODO: filename arg
    # TODO: quiet mode all over or mutually exclusively with filename
    # plt.pcolormesh(binx, biny, d_1_2, alpha=0.5)

    # couldn't even try out the , shading="gouraud" due to a
    # "Dimensions of C (99, 99) are incompatible with X (100) and/or Y (100);
    # see help(pcolormesh)" probably due to
    # https://github.com/matplotlib/matplotlib/issues/8422
    #
    # alpha=0.5 otherwise looks horrible unfortunately

    # cmap1 = plt.get_cmap('PiYG')
    div = np.maximum(np.max(d_2_1), np.max(d_1_2)) * 1.3

    cmap1 = plt.get_cmap('copper')
    cmap2 = plt.get_cmap('bone')

    plt.pcolormesh(binx, biny, d_1_2, cmap=cmap1, norm=Normalize(vmax=div))
    d_2_1 = np.ma.masked_array( d_2_1, d_2_1 == 0)
    plt.pcolormesh(binx, biny, d_2_1, cmap=cmap2, norm=Normalize(vmax=div))

    if filename:
        plt.savefig(filename)

    if show:
        plt.show()


def plot_2014_fingerprints(parties=PARTIES_2014):
    # in 2014 there was nomentum lol
    df_2014 = get_2014_cleaned_data()

    for party_2014 in parties:

        df_2014_top_90 = df_2014[
            df_2014.Telepules.isin(_ranking.iloc[:90].Telepules)
        ]

        df_2014_top_91_to_bottom = df_2014[
            df_2014.Telepules.isin(_ranking.iloc[90:].Telepules)
        ]

        plot_fingerprint(df_2014_top_91_to_bottom[party_2014],
                         df_2014_top_91_to_bottom["Ervenyes"],
                         df_2014_top_91_to_bottom["Nevjegyzekben"],
                         "2014 least suspicious",
                         "Figure_2014_%s_top_91_to_bottom.png" % party_2014,
                         zoom_onto=party_2014 in ZOOM_ONTO)
        plot_fingerprint(df_2014_top_90[party_2014],
                         df_2014_top_90["Ervenyes"],
                         df_2014_top_90["Nevjegyzekben"],
                         "2014 most suspicious",
                         "Figure_2014_%s_top_90.png" % party_2014,
                         zoom_onto=party_2014 in ZOOM_ONTO)


def plot_2018_fingerprints(parties=PARTIES_2018):

    df_2018 = get_2018_cleaned_data()

    for party_2018 in parties:

        df_2018_top_90 = df_2018[
            df_2018.Telepules.isin(_ranking.iloc[:90].Telepules)
        ]
        df_2018_top_91_to_bottom = df_2018[
            df_2018.Telepules.isin(_ranking.iloc[90:].Telepules)
        ]
        plot_fingerprint(df_2018_top_91_to_bottom[party_2018],
                         df_2018_top_91_to_bottom["Ervenyes"],
                         df_2018_top_91_to_bottom["Nevjegyzekben"],
                         "2018 least suspicious",
                         "Figure_2018_%s_top_91_to_bottom.png" % party_2018,
                         zoom_onto=party_2018 in ZOOM_ONTO)
        plot_fingerprint(df_2018_top_90[party_2018],
                         df_2018_top_90["Ervenyes"],
                         df_2018_top_90["Nevjegyzekben"],
                         "2018 most suspicious",
                         "Figure_2018_%s_top_90.png" % party_2018,
                         zoom_onto=party_2018 in ZOOM_ONTO)


def plot_2019_fingerprints(parties=PARTIES_2019):

    df_2019 = get_preprocessed_data()

    for party_2019 in parties:

        df_2019_top_90 = df_2019[
            df_2019.Telepules.isin(_ranking.iloc[:90].Telepules)
        ]
        df_2019_top_91_to_bottom = df_2019[
            df_2019.Telepules.isin(_ranking.iloc[90:].Telepules)
        ]
        plot_fingerprint(df_2019_top_91_to_bottom[party_2019],
                         df_2019_top_91_to_bottom["Ervenyes"],
                         df_2019_top_91_to_bottom["Nevjegyzekben"],
                         "2019 least suspicious",
                         "Figure_2019_%s_top_91_to_bottom.png" % party_2019,
                         zoom_onto=party_2019 in ZOOM_ONTO)
        plot_fingerprint(df_2019_top_90[party_2019],
                         df_2019_top_90["Ervenyes"],
                         df_2019_top_90["Nevjegyzekben"],
                         "2019 most suspicious",
                         "Figure_2019_%s_top_90.png" % party_2019,
                         zoom_onto=party_2019 in ZOOM_ONTO)


def check_fingerprint_diff(df, party, show=True, filename=None):

    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/quadmesh_demo.html
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    df_top_90 = df[
        df.Telepules.isin(_ranking.iloc[:90].Telepules)
    ]
    df_top_91_to_bottom = df[
        df.Telepules.isin(_ranking.iloc[90:].Telepules)
    ]

    def votes_to_coords(df, party):
        # x: turnout, y: winning vote proportion, weight#
        # looks like it needs to be transposed when playing with np.histogram2d
        return (df[party] / df["Ervenyes"],
                df["Ervenyes"] / df["Nevjegyzekben"],
                df[party])

    x1, y1, w1 = votes_to_coords(df_top_91_to_bottom, party)
    x2, y2, w2 = votes_to_coords(df_top_90, party)
    s1 = sum(df_top_90[party])
    s2 = sum(df_top_91_to_bottom[party])
    print("votes won when not susp", s2, "susp", s1, "ratio", s1 / s2)
    df1 = df_top_90
    df2 = df_top_91_to_bottom
    print("mean valid votes:",
          np.mean(df1["Ervenyes"]),
          np.mean(df2["Ervenyes"]))
    print("total valid votes ratio \"achieved:",
          np.sum(df1["Ervenyes"]) /
          np.sum(df2["Ervenyes"]))
    print("vote ratio in susp. areas %.2f" % (s1 / sum(df1["Ervenyes"]) * 100))
    print("additional votes perc. in more susp. areas %.2f %%" %
          ((s1 - s2) / (sum(df1["Ervenyes"]) - sum(df2["Ervenyes"])) * 100))
    # have to go CLT
    #
    # std_indiv / sqrt(n_sampl)   ~   sd of the mean
    # std_indiv * sqrt(n_sampl)   ~   sd of the sum
    # sd of the sum over the sum  ~   rel. uncertainty in nr. of votes
    print("rel. sd ratio (< 1 for suspect targeted manipulation):",
          (np.std(df1[party] * (len(df1) ** 0.5)) / np.sum(df1[party])) /
          (np.std(df2[party] * (len(df2) ** 0.5)) / np.sum(df2[party]))
          )

    d_1_2, d_2_1, binx, biny = get_diffs(x1, y1, w1, x2, y2, w2)
    plot_histogram2d(d_1_2, d_2_1, binx, biny, show, filename)



def plot_fingerprint_diffs(show: bool):
    print("Fingerprint differences")

    def filename(year, party):
        return os.path.join(FINGERPRINT_DIR,
                            "diff_%d_%s.png" % (year, party))

    df_2014 = get_2014_cleaned_data()
    for party_2014 in PARTIES_2014:
        print("2014", party_2014)
        check_fingerprint_diff(df_2014, party_2014,
                               show=show,
                               filename=filename(2014, party_2014))

    df_2018 = get_2018_cleaned_data()
    for party_2018 in PARTIES_2018:
        print("2018", party_2018)
        check_fingerprint_diff(df_2018, party_2018,
                               show=show,
                               filename=filename(2018, party_2018))

    df_2019 = get_preprocessed_data()
    for party_2019 in PARTIES_2019:
        print("2019", party_2019)
        check_fingerprint_diff(df_2019, party_2019,
                               show=show,
                               filename=filename(2019, party_2019))


if __name__ == "__main__":
    if not os.path.exists(FINGERPRINT_DIR):
        os.mkdir(FINGERPRINT_DIR)
    plot_2014_fingerprints()
    plot_2018_fingerprints()
    plot_2019_fingerprints()
    plot_fingerprint_diffs(show=False)
