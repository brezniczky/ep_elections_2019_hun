from preprocessing import get_preprocessed_data
from matplotlib.colors import Normalize
from cleaning import (get_2010_cleaned_data,
                      get_2014_cleaned_data,
                      get_2018_cleaned_data)
import matplotlib.pyplot as plt
import numpy as np
import os
import app15_overall_ranking as app15

# todo: 1. an insane amount of refactoring :) :(


_ranking = app15.get_overall_list_last_2_years()


FINGERPRINT_DIR = "fingerprints"


PARTIES_2010 = ["Fidesz-KDNP", "Jobbik", "LMP", 'MSZP', "MDF", "MIÉP",
                "MSZDP"]  # a few were left off here too


PARTIES_2014 = ["Fidesz-KDNP", "Jobbik", "LMP", "Együtt 2014",
                'MSZP-Együtt-DK-PM-MLP']

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


PARTY_2019_TO_2018_DICT = {
    "Fidesz": "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT",
    "Jobbik": 'JOBBIK MAGYARORSZÁGÉRT MOZGALOM',
    "LMP": 'LEHET MÁS A POLITIKA',
    "Momentum": 'MOMENTUM MOZGALOM',
    "DK": 'DEMOKRATIKUS KOALÍCIÓ',
    "MSZP": "MAGYAR SZOCIALISTA PÁRT-PÁRBESZÉD MAGYARORSZÁGÉRT PÁRT"
}


PARTY_2019_TO_2014_DICT = {
    "Fidesz": "Fidesz-KDNP",
    "Jobbik": 'Jobbik',
    "LMP": 'LMP',
    "Momentum": None,
    "DK": 'MSZP-Együtt-DK-PM-MLP',
    # "MSZP": 'MSZP-Együtt-DK-PM-MLP'
}


PARTY_2019_TO_2010_DICT = {
    "Fidesz": "Fidesz-KDNP",
    "Jobbik": 'Jobbik',
    "LMP": 'LMP',
    "Momentum": None,
    "DK": None,
    "MSZP": "MSZP"
}


ZOOM_ONTO = [
    "Együtt 2014", 'EGYÜTT - A KORSZAKVÁLTÓK PÁRTJA'
]


# it was read by the eye - could be obtained by e.g. clustering in the long run
SUSPECT_CENTROID_POS_TURNOUT_AND_WINNER_RATE = [0.63, 0.455]
SUSPECT_CENTROID_X_RAD = 0.08
SUSPECT_CENTROID_Y_RAD = 0.07


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
    full_filename = os.path.join(FINGERPRINT_DIR, filename)
    plt.savefig(full_filename)
    print("plot saved as %s" % full_filename)
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


def plot_fingerprints_for_year(parties, year):
    df_functions = {
        2010: get_2010_cleaned_data,
        2014: get_2014_cleaned_data,
        2018: get_2018_cleaned_data,
        2019: get_preprocessed_data,
    }
    df = df_functions[year]()
    for party in parties:
        df_top_90 = df[
            df.Telepules.isin(_ranking.iloc[:90].Telepules)
        ]
        df_top_91_to_bottom = df[
            df.Telepules.isin(_ranking.iloc[90:].Telepules)
        ]

        plot_fingerprint(df_top_91_to_bottom[party],
                         df_top_91_to_bottom["Ervenyes"],
                         df_top_91_to_bottom["Nevjegyzekben"],
                         "%d least suspicious" % year,
                         "Figure_%d_%s_top_91_to_bottom.png" %
                         (year, party),
                         zoom_onto=party in ZOOM_ONTO)
        plot_fingerprint(df_top_90[party],
                         df_top_90["Ervenyes"],
                         df_top_90["Nevjegyzekben"],
                         "%d most suspicious" % year,
                         "Figure_%d_%s_top_90.png" % (year, party),
                         zoom_onto=party in ZOOM_ONTO)


def plot_2010_fingerprints(parties=PARTIES_2010):
    plot_fingerprints_for_year(parties, 2010)


def plot_2014_fingerprints(parties=PARTIES_2014):
    plot_fingerprints_for_year(parties, 2014)


def plot_2018_fingerprints(parties=PARTIES_2018):
    plot_fingerprints_for_year(parties, 2018)


def plot_2019_fingerprints(parties=PARTIES_2019):
    plot_fingerprints_for_year(parties, 2019)


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


def list_suspects_near_2019_fingerprint(point, r_x, r_y, filter=True):
    df = get_preprocessed_data()
    df = df[df.Telepules.isin(_ranking.iloc[:90].Telepules)]
    df["Turnout"] = df["Ervenyes"] / df["Nevjegyzekben"]
    df["Fidesz_rate"] = df["Fidesz"] / df["Ervenyes"]
    df["In_centroid"] = (((df["Turnout"] - point[0]) / r_x) ** 2 +
                         ((df["Fidesz_rate"] - point[1]) / r_y) ** 2) ** 0.5 < 1
    df.sort_values(["Telepules", "Szavazokor"], inplace=True)
    if filter:
        df =  df[df["In_centroid"]]
        df[["Telepules", "Szavazokor"]].to_csv(
            "app16_fingerprint_suspects_2019.csv",
            index=False
        )
    return df


def plot_municipality(municipality_str="Budapest I.", party="Fidesz", year=2019,
                      startswith=True,
                      translate_party_name=True,
                      return_coords=False,
                      highlight_last_digit=None):
    if year == 2019:
        df = get_preprocessed_data()
    elif year == 2018:
        df = get_2018_cleaned_data()
        if translate_party_name:
            party = PARTY_2019_TO_2018_DICT[party]
    elif year == 2014:
        df = get_2014_cleaned_data()
        if translate_party_name:
            party = PARTY_2019_TO_2014_DICT[party]
    elif year == 2010:
        df = get_2010_cleaned_data()
        if translate_party_name:
            party = PARTY_2019_TO_2010_DICT[party]

    if startswith:
        df = df[df.Telepules.str.startswith(municipality_str)]
        if len(set(df["Telepules"].values)) > 1:
            raise Exception(
                "Muncipality pattern is not unequivocal.\n"
                "Please provide a more specific string or set startswith=False."
            )
    else:
        df = df[df.Telepules==municipality_str]

    municipality = set(df["Telepules"].values)
    if not municipality:
        print("Municipality string %s was not found." % municipality_str)
        exit()

    municipality = list(municipality)[0]

    print("total %s/total Ervenyes %.2f %%" %
          (party, sum(df[party]) / sum(df["Ervenyes"]) * 100))
    x = df["Ervenyes"] / df["Nevjegyzekben"]
    y = df[party] / df["Ervenyes"]
    plt.scatter(x, y)

    title = "%s %s %s" % (year, party, municipality)
    if highlight_last_digit is not None:
        is_digit = (df[party] % 10 == highlight_last_digit)  # |
                    # (df["Ervenyes"] % 10 == highlight_last_digit))
        xh = x[is_digit]
        yh = y[is_digit]
        plt.scatter(xh, yh)
        title = "%s highlighting digit %s" % (title, highlight_last_digit)
        print("digit dist:", np.unique(
            # list(df["Ervenyes"] % 10) +
            list(df[party] % 10),
            return_counts=True))

    for ax, ay, award in zip(x, y, df["Szavazokor"]):
        plt.annotate(int(award), (ax + 0.005, ay))

    plt.title(title)

    plt.show()
    if return_coords:
        return x, y


if __name__ == "__main__":
    if not os.path.exists(FINGERPRINT_DIR):
        os.mkdir(FINGERPRINT_DIR)
    plot_2010_fingerprints()
    plot_2014_fingerprints()
    plot_2018_fingerprints()
    plot_2019_fingerprints()

    plot_fingerprint_diffs(show=False)
    df_suspect = (list_suspects_near_2019_fingerprint(
        SUSPECT_CENTROID_POS_TURNOUT_AND_WINNER_RATE,
        SUSPECT_CENTROID_X_RAD,
        SUSPECT_CENTROID_Y_RAD
    ))
    # plot_municipality("Miskolc", "Fidesz", 2019, highlight_last_digit=0)
    # plot_municipality("Miskolc", "Fidesz", 2019, highlight_last_digit=5)
    # plot_municipality("Miskolc", "Fidesz", 2019, highlight_last_digit=7)
    # plot_municipality("Miskolc", "Fidesz", 2014)
    # plot_municipality("Szeged", 'MSZP', 2018, highlight_last_digit=None)
