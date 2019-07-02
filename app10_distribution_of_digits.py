"""
A few plots about
- the digit distributions by party
- the voter count distributions by digit and party
- how the threshold affects the digit distribution
  (spoiler alert: apparently not at all)
TODO:
- ranking of parties by entropy of vote digits for valid votes >= ...
"""
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import get_preprocessed_data


def plot_votes_of_digits_hist(df, party, digit_groups, n_bins=20,
                              max_votes=None, title=None):
    if max_votes is None:
        max_votes = max(df[party])
    bin_size = max_votes / n_bins
    # so that an equal number of identical last digits fall in each bucket
    bin_size = max(10, round(bin_size / 10) * 10)

    bins = np.arange(0, max_votes, bin_size)
    alpha = 1 / len(digit_groups)
    for digit_group in digit_groups:
        plt.hist(df[df["ld_" + party].isin(digit_group)][party],
                 alpha=alpha, bins=bins)
    if title is None:
        title = party
    plt.title(title)


def plot_9_to_7_digit_distributions():
    df = get_preprocessed_data()

    # f, ax = plt.subplots(4, 3)
    f = plt.figure()
    f.suptitle("Distribution of number of votes ending in 9s vs. 7s, 2019 values")

    i = 1

    def plot_to_next(l):
        nonlocal i
        f.add_subplot(4, 3, i)
        i += 1
        l()
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Nevjegyzekben", [[9], [7]], 50, 2000))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Ervenyes", [[9], [7]], 50, 2000))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Ervenytelen", [[9], [7]], 50))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Fidesz", [[9], [7]], 30, 600))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "DK", [[9], [7]]))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Momentum", [[9], [7]], 30, 250))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Jobbik", [[9], [7]]))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "Mi Hazank", [[9], [7]]))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "LMP", [[9], [7]], 30, 250))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "MSZP", [[9], [7]]))
    plot_to_next(lambda: plot_votes_of_digits_hist(df, "MKKP", [[9], [7]]))
    plt.tight_layout()
    plt.show()


def plot_party_vote_by_digit_relationships(party="Fidesz", max_votes=None):
    df = get_preprocessed_data()

    # f, ax = plt.subplots(4, 3)
    f = plt.figure()
    f.suptitle("Distribution of %s vote counts ending in "
               "7s vs. others, 2019 values" % party)
    i = 1

    def plot_to_next(l):
        nonlocal i
        f.add_subplot(3, 3, i)
        i += 1
        l()
    for digit in range(10):
        if digit != 7:
            plot_to_next(
                lambda: plot_votes_of_digits_hist(
                    df, party, [[digit], [7]], 30,
                    title="%d" % digit,
                    max_votes=max_votes
                ),
            )
    plt.tight_layout()
    plt.show()


def plot_Fidesz_digits_sensitivity_to_cutoff():
    fig = plt.figure()
    # cutoffs at say 100 can introduce a disproportionate amount of zeroes
    # one way to eliminate such worries
    fig.add_subplot(1, 2, 1)
    plt.title(u"Fidesz votes \u2265 100 only")
    df = get_preprocessed_data()
    np.random.seed(5)
    for i in range(10):
        plt.hist(df.ld_Fidesz[df.Fidesz >= 95 + np.random.choice(range(11),
                 len(df))], bins=10, alpha=0.1)

    df = get_preprocessed_data()
    big_enough_ones = df[["Telepules"]].groupby(["Telepules"]).aggregate({"Telepules": len})
    big_enough_ones.columns = ["n_wards"]
    big_enough_ones.reset_index(inplace=True)
    big_enough_ones = set(big_enough_ones.Telepules)

    fig.add_subplot(1, 2, 2)
    np.random.seed(5)
    plt.title(u"With ward count \u2265 8")
    for i in range(10):
        cond1 = df.Fidesz >= 95 + np.random.choice(range(11), len(df))
        cond2 = df.Telepules.isin(big_enough_ones)
        plt.hist(df.ld_Fidesz[cond1 & cond2], bins=10, alpha=0.1)
    plt.show()


def plot_digit_distributions():
    df = get_preprocessed_data()

    f = plt.figure()
    f.suptitle("Last digit distributions, 2019 values")
    parties = ["Nevjegyzekben", "Ervenyes", "Ervenytelen",
               "Fidesz", "DK", "Jobbik",
               "Mi Hazank", "Momentum", "MSZP",
               "LMP", "MKKP"]
    prev = None
    for idx in range(len(parties)):
        ax = f.add_subplot(3, 4, idx + 1, ymargin=0.5, sharey=prev)
        prev = ax
        ax.get_xaxis().set_visible(False)
        party = parties[idx]
        plt.hist(df["ld_" + party], bins=range(11))

        plt.title(party)
    plt.show()


if __name__ == "__main__":
    plot_party_vote_by_digit_relationships("Fidesz", max_votes=600)
    plot_party_vote_by_digit_relationships("DK", max_votes=300)

    plot_party_vote_by_digit_relationships("Momentum", max_votes=300)
    plot_party_vote_by_digit_relationships("Jobbik", max_votes=300)
    plot_party_vote_by_digit_relationships("LMP", max_votes=50)
    plot_party_vote_by_digit_relationships("MSZP", max_votes=50)
;