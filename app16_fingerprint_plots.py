from preprocessing import get_preprocessed_data
from cleaning import get_2014_cleaned_data, get_2018_cleaned_data
import matplotlib.pyplot as plt
import numpy as np
import app15_overall_ranking as app15


_ranking = app15.get_overall_list_last_2_years()


def plot_fingerprint(winner_votes, valid_votes,
                     registered_voters, title, filename,
                     weighted=True):

    bins = [np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)]

    weights = None if not weighted else winner_votes
    plt.hist2d(
        # winner_votes / registered_voters,  # TODO: or valid_votes?
        valid_votes / registered_voters,
        winner_votes / valid_votes,  # TODO: or valid_votes?
        bins=bins,
        weights=weights
    )
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def plot_2014_fingerprints():
    party_2014 = "Fidesz-KDNP"
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
                     "Figure_2014_top_91_to_bottom.png")
    plot_fingerprint(df_2014_top_90[party_2014],
                     df_2014_top_90["Ervenyes"],
                     df_2014_top_90["Nevjegyzekben"],
                     "2014 most suspicious",
                     "Figure_2014_top_90.png")


def plot_2018_fingerprints():
    party_2018 = "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT"
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
                     "Figure_2018_top_91_to_bottom.png")
    plot_fingerprint(df_2018_top_90[party_2018],
                     df_2018_top_90["Ervenyes"],
                     df_2018_top_90["Nevjegyzekben"],
                     "2018 most suspicious",
                     "Figure_2018_top_90.png")


def plot_2019_fingerprints():
    party_2019 = "Fidesz"
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
                     "Figure_2019_top_91_to_bottom.png")
    plot_fingerprint(df_2019_top_90[party_2019],
                     df_2019_top_90["Ervenyes"],
                     df_2019_top_90["Nevjegyzekben"],
                     "2019 most suspicious",
                     "Figure_2019_top_90.png")



if __name__ == "__main__":
    plot_2014_fingerprints()
