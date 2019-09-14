from HU.preprocessing import get_preprocessed_data
from HU.cleaning import (get_2010_cleaned_data,
                         get_2014_cleaned_data,
                         get_2018_cleaned_data)
from HU.fingerprint_plots import plot_fingerprint_diff
from drdigit import plot_fingerprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import HU.app15_overall_ranking as app15
from arguments import save_output, is_quiet, get_output_dir, is_quick

# todo: 1. a degree amount of refactoring :) :(


DOWNSAMPLE_SEED = 1234
N_TOP = 90
CHECK_DOWNSAMPLED = False


FINGERPRINT_DIR = os.path.join(get_output_dir(), "fingerprints")


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


_ranking = None


def _get_ranking():
    global _ranking

    if _ranking is None:
        _ranking = app15.get_overall_list_last_2_years()
        if CHECK_DOWNSAMPLED:
            print("setting random seed for testing with downsampled rankings")
            np.random.seed(DOWNSAMPLE_SEED)
            _ranking = _ranking.iloc[
                sorted(np.random.choice(range(len(_ranking)), N_TOP * 2,
                                        # this breaks away from bootstrapping
                                        # but feels okay
                                        replace=False))]
    return _ranking


def _get_df(year):
    df_functions = {
        2010: get_2010_cleaned_data,
        2014: get_2014_cleaned_data,
        2018: get_2018_cleaned_data,
        2019: get_preprocessed_data,
    }
    return df_functions[year]()


def plot_fingerprints_for_year(parties, year, save, reduce_party_name=True,
                               plot_comparative=True):

    def reduce_name(s):
        if reduce_party_name and len(s) > 20:
            s = s[:20] + "..."
        return s

    df = _get_df(year)
    for party in parties:
        top_municipalities = _get_ranking().iloc[:N_TOP].Telepules
        bottom_municipalities = _get_ranking().iloc[N_TOP:].Telepules
        df_top_90 = df[df.Telepules.isin(top_municipalities)]
        df_top_91_to_bottom = df[df.Telepules.isin(bottom_municipalities)]

        plot_fingerprint(df_top_91_to_bottom[party],
                         df_top_91_to_bottom["Ervenyes"],
                         df_top_91_to_bottom["Nevjegyzekben"],
                         title="%d %s least suspicious" % (year,
                                                           reduce_name(party)),
                         filename=("Figure_%d_%s_top_91_to_bottom.png" %
                                   (year, party))
                                   if save else None,
                         zoom_onto=party in ZOOM_ONTO,
                         fingerprint_dir=FINGERPRINT_DIR,
                         quiet=is_quiet())
        plot_fingerprint(df_top_90[party],
                         df_top_90["Ervenyes"],
                         df_top_90["Nevjegyzekben"],
                         title="%d %s most suspicious" %
                               (year, reduce_name(party)),
                         filename=("Figure_%d_%s_top_90.png" % (year, party))
                                  if save else None,
                         zoom_onto=party in ZOOM_ONTO,
                         fingerprint_dir=FINGERPRINT_DIR,
                         quiet=is_quiet())
        if plot_comparative:
            plot_fingerprint_diff(df, party,
                                  top_municipalities, bottom_municipalities,
                                  show=not is_quiet(),
                                  filename=("Figure_%d_%s_diff.png" %
                                            (year, party))
                                            if save else None)
        if is_quick():
            break


def plot_2010_fingerprints(parties=PARTIES_2010, save=True):
    plot_fingerprints_for_year(parties, 2010, save)


def plot_2014_fingerprints(parties=PARTIES_2014, save=True):
    plot_fingerprints_for_year(parties, 2014, save)


def plot_2018_fingerprints(parties=PARTIES_2018, save=True):
    plot_fingerprints_for_year(parties, 2018, save)


def plot_2019_fingerprints(parties=PARTIES_2019, save=True):
    plot_fingerprints_for_year(parties, 2019, save)


def plot_fingerprint_diffs(show: bool):
    print("Fingerprint differences")
    top_municipalities = _get_ranking().iloc[:N_TOP].Telepules
    bottom_municipalities = _get_ranking().iloc[N_TOP:].Telepules

    def filename(year, party):
        return os.path.join(FINGERPRINT_DIR,
                            "diff_%d_%s.png" % (year, party))

    df_2010 = get_2010_cleaned_data()
    for party_2010 in PARTIES_2010:
        print("2010", party_2010)
        plot_fingerprint_diff(df_2010, party_2010,
                              show=show,
                              filename=filename(2010, party_2010),
                              top_municipalities=top_municipalities,
                              bottom_municipalities=bottom_municipalities)
        if is_quick():
            return

    df_2014 = get_2014_cleaned_data()
    for party_2014 in PARTIES_2014:
        print("2014", party_2014)
        plot_fingerprint_diff(df_2014, party_2014,
                              show=show,
                              filename=filename(2014, party_2014),
                              top_municipalities=top_municipalities,
                              bottom_municipalities=bottom_municipalities)

    df_2018 = get_2018_cleaned_data()
    for party_2018 in PARTIES_2018:
        print("2018", party_2018)
        plot_fingerprint_diff(df_2018, party_2018,
                              show=show,
                              filename=filename(2018, party_2018),
                              top_municipalities=top_municipalities,
                              bottom_municipalities=bottom_municipalities)

    df_2019 = get_preprocessed_data()
    for party_2019 in PARTIES_2019:
        print("2019", party_2019)
        plot_fingerprint_diff(df_2019, party_2019,
                              show=show,
                              filename=filename(2019, party_2019),
                              top_municipalities=top_municipalities,
                              bottom_municipalities=bottom_municipalities)


def _select_2019_prime_suspect_wards(df, point, r_x, r_y):
    df = df[df.Telepules.isin(_get_ranking().iloc[:N_TOP].Telepules)]
    df_saved = df  # to prevent overwriting values etc.
    df = df.copy()
    df["Turnout"] = df["Ervenyes"] / df["Nevjegyzekben"]
    df["Fidesz_rate"] = df["Fidesz"] / df["Ervenyes"]
    df["In_centroid"] = (((df["Turnout"] - point[0]) / r_x) ** 2 +
                         ((df["Fidesz_rate"] - point[1]) / r_y) ** 2) ** 0.5 < 1
    return df_saved[df["In_centroid"]]


def list_suspects_near_2019_fingerprint(point, r_x, r_y, save_generated=True):
    df = get_preprocessed_data()
    df = _select_2019_prime_suspect_wards(df, point, r_x, r_y)

    df.sort_values(["Telepules", "Szavazokor"], inplace=True)
    if save_generated:
        save_output(
            df[["Telepules", "Szavazokor"]],
            "app16_fingerprint_suspects_2019.csv",
        )
    return df


def _translate_party_name(party, year):
    dicts = {
        2018: PARTY_2019_TO_2018_DICT,
        2014: PARTY_2019_TO_2014_DICT,
        2010: PARTY_2019_TO_2010_DICT,
    }
    return dicts[year][party] if year != 2019 else party


def _select_municipality(df, municipality_str, startswith):
    # municipality: have to rename this to city - who can type this ...
    if startswith:
        df = df[df.Telepules.str.startswith(municipality_str)]
        if len(set(df["Telepules"].values)) > 1:
            raise Exception(
                "Municipality pattern is not unequivocal.\n"
                "Please provide a more specific string or set startswith=False."
            )
    else:
        df = df[df.Telepules==municipality_str]

    municipality = set(df["Telepules"].values)
    if not municipality:
        print("Municipality string %s was not found." % municipality_str)
        exit()

    municipality = list(municipality)[0]

    return df, municipality


_suspect_2019_wards = None

def _get_suspect_2019_wards():
    global _suspect_2019_wards
    if _suspect_2019_wards is None:
        _suspect_2019_wards = list_suspects_near_2019_fingerprint(
            SUSPECT_CENTROID_POS_TURNOUT_AND_WINNER_RATE,
            SUSPECT_CENTROID_X_RAD, SUSPECT_CENTROID_Y_RAD,
            save_generated=False
        )
    return _suspect_2019_wards


def _select_suspicious_in_2019(df):
    df_2 = _get_suspect_2019_wards()
    # TODO: data type compat. should be ensured on load
    df["Szavazokor"] = df["Szavazokor"].astype(int)
    df_2["Szavazokor"] = df_2["Szavazokor"].astype(int)
    df = pd.merge(df, df_2[["Telepules", "Szavazokor"]])
    return df


def plot_municipality(municipality_str="Budapest I.", party="Fidesz", year=2019,
                      municipality_startswith=True,
                      translate_party_name=True,
                      return_coords=False,
                      highlight_last_digit=None,
                      highlight_suspicious=False):
    """
    Create a scatter plot of the ward specific results for the municipality.

    :param municipality_str: name (may be the beginning) to seek
    :param party:
    :param year:
    :param municipality_startswith:
    :param translate_party_name:
    :param return_coords:
    :param highlight_last_digit:
    :param highlight_suspicious:
    :return:
    """
    if translate_party_name:
        party = _translate_party_name(party, year)

    df = _get_df(year)

    df, municipality = _select_municipality(df, municipality_str, municipality_startswith)

    print("total %s/total Ervenyes %.2f %%" %
          (party, sum(df[party]) / sum(df["Ervenyes"]) * 100))
    x = df["Ervenyes"] / df["Nevjegyzekben"]
    y = df[party] / df["Ervenyes"]
    plt.scatter(x, y)

    title = "%s %s %s" % (year, party, municipality)
    for ax, ay, award in zip(x, y, df["Szavazokor"]):
        plt.annotate(int(award), (ax + 0.005, ay))

    if highlight_last_digit is not None:
        is_digit = (df[party] % 10 == highlight_last_digit)  # |
                    # (df["Ervenyes"] % 10 == highlight_last_digit))
        xh = df["x"][is_digit]
        yh = df["y"][is_digit]
        plt.scatter(xh, yh)
        title = "%s highlighting digit %s" % (title, highlight_last_digit)
        print("digit dist:", np.unique(
            # list(df["Ervenyes"] % 10) +
            list(df[party] % 10),
            return_counts=True))

    if highlight_suspicious:
        df["x"] = x
        df["y"] = y
        df = _select_suspicious_in_2019(df)
        plt.scatter(df["x"], df["y"])

    plt.title(title)

    plt.show()
    if return_coords:
        return x, y


if __name__ == "__main__":
    if not os.path.exists(FINGERPRINT_DIR):
        os.mkdir(FINGERPRINT_DIR)
    if not is_quick():
        plot_2010_fingerprints()
        plot_2014_fingerprints()
        plot_2018_fingerprints()
    plot_2019_fingerprints()

    plot_fingerprint_diffs(show=not is_quiet())

    df_suspect = (list_suspects_near_2019_fingerprint(
        SUSPECT_CENTROID_POS_TURNOUT_AND_WINNER_RATE,
        SUSPECT_CENTROID_X_RAD,
        SUSPECT_CENTROID_Y_RAD,
        save_generated=True
    ))

    # plot_municipality("Miskolc", "Fidesz", 2019, highlight_last_digit=0)
    # plot_municipality("Miskolc", "Fidesz", 2019, highlight_last_digit=5)
    # plot_municipality("Miskolc", "Fidesz", 2019, highlight_last_digit=7)
    # plot_municipality("Miskolc", "Fidesz", 2014)
    # plot_municipality("Szeged", 'MSZP', 2018, highlight_last_digit=None)
    # plot_municipality("Budapest III", 'Fidesz', 2019, highlight_last_digit=None, suspicious_only=True)

    # plot_municipality("Budapest III", 'Fidesz', 2019, highlight_last_digit=None,
    #                   highlight_suspicious=True)
    # plot_municipality("Budapest III", 'Fidesz', 2018, highlight_suspicious=True)
    # plot_municipality("Budapest III", 'Fidesz', 2014, highlight_suspicious=True)
    # plot_municipality("Budapest III", 'Fidesz', 2010, highlight_suspicious=True)
