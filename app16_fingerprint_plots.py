from preprocessing import get_preprocessed_data
from cleaning import get_2014_cleaned_data, get_2018_cleaned_data
import matplotlib.pyplot as plt
import numpy as np
import os
import app15_overall_ranking as app15


_ranking = app15.get_overall_list_last_2_years()


FINGERPRINT_DIR = "fingerprints"


PARTIES_2014 = ["Fidesz-KDNP", "Jobbik", "LMP", "Együtt 2014",
                'MSZP-Együtt-DK-PM-MLP']

PARTIES_2018 = [
    "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT",
    'JOBBIK MAGYARORSZÁGÉRT MOZGALOM',
    'LEHET MÁS A POLITIKA',
    'MOMENTUM MOZGALOM',
    'DEMOKRATIKUS KOALÍCIÓ',
    'EGYÜTT - A KORSZAKVÁLTÓK PÁRTJA'
]

PARTIES_2019 = ["Fidesz", "Jobbik", "LMP", "Momentum", "DK"]

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


def plot_2014_fingerprints(parties=PARTIES_2014):
    # in 2014 there was nomentum lol

    for party_2014 in parties:
        df_2014 = get_2014_cleaned_data()

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
    for party_2018 in parties:
        df_2018 = get_2018_cleaned_data()

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
    for party_2019 in parties:
        df_2019 = get_preprocessed_data()

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


if __name__ == "__main__":
    if not os.path.exists(FINGERPRINT_DIR):
        os.mkdir(FINGERPRINT_DIR)
    plot_2014_fingerprints()
    plot_2018_fingerprints()
    plot_2019_fingerprints()
