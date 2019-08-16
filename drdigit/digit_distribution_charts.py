import matplotlib.pyplot as plt
import numpy as np


USE_MEAN_DIGIT_GROUP = 4.5


def _plot_votes_of_digits_hist(df, party, digit_groups, n_bins=20,
                               max_votes=None, title=None,
                               last_digit_colname_prefix="ld_"):

    """ Special digit group value: 4.5, meaning "mean of all". """

    if max_votes is None:
        max_votes = max(df[party])
    bin_size = max_votes / n_bins
    # so that an equal number of each last digit fall in each bucket in a
    # uniform case
    bin_size = max(10, round(bin_size / 10) * 10)

    bins = np.arange(0, max_votes, bin_size)
    alpha = 1 / len(digit_groups)
    for digit_group in digit_groups:
        if digit_group != USE_MEAN_DIGIT_GROUP:
            plt.hist(df[df[last_digit_colname_prefix +
                           party].isin(digit_group)][party],
                     alpha=alpha, bins=bins)
        else:
            plt.hist(df[party],
                     weights=[0.1] * len(df), alpha=alpha, bins=bins)
    if title is None:
        title = party
    plt.title(title)


def _plot_party_vote_by_digit_relationships_with_average(
        df, party, max_votes, last_digit_colname_prefix, n_bins
):
    f = plt.figure()
    f.suptitle("Distribution of %s vote counts by last digit vs. their average "
               "empiric distribution" % (party))
    i = 1

    def plot_to_next(l):
        nonlocal i
        f.add_subplot(4, 3, i)
        i += 1
        l()

    for digit in range(10):
        plot_to_next(
            lambda: _plot_votes_of_digits_hist(
                df, party, [[digit], 4.5], n_bins=n_bins,
                title="%d" % digit,
                max_votes=max_votes,
                last_digit_colname_prefix=last_digit_colname_prefix
            ),
        )
    plt.tight_layout()
    plt.show()


def plot_party_vote_by_digit_relationships(df, party="Fidesz", ref_digit=7,
                                           max_votes=None,
                                           last_digit_colname_prefix="ld_",
                                           n_bins=20):
    """

    :param df: columns should contain
    :param party:
    :param ref_digit:
    :param max_votes:
    :return:
    """
    assert ref_digit in list(range(0, 10)) + [USE_MEAN_DIGIT_GROUP]
    if ref_digit == USE_MEAN_DIGIT_GROUP:
        return _plot_party_vote_by_digit_relationships_with_average(
            df, party, max_votes, last_digit_colname_prefix, n_bins
        )

    f = plt.figure()
    f.suptitle("Distribution of %s vote counts ending in "
               "%ds vs. others, 2019 values" % (party, ref_digit))
    i = 1

    def plot_to_next(l):
        nonlocal i
        f.add_subplot(3, 3, i)
        i += 1
        l()

    for digit in range(10):
        if digit != ref_digit:
            plot_to_next(
                lambda: _plot_votes_of_digits_hist(
                    df, party, [[digit], [ref_digit]], n_bins=n_bins,
                    title="%d" % digit,
                    max_votes=max_votes,
                    last_digit_colname_prefix=last_digit_colname_prefix
                ),
            )
    plt.tight_layout()
    plt.subplots_adjust(top=0.875)
    plt.show()


if __name__ == "__main__":
    from preprocessing import get_preprocessed_data
    df = get_preprocessed_data()
    # plot_party_vote_by_digit_relationships(df, "Fidesz", max_votes=600)
    # plot_party_vote_by_digit_relationships(df, "DK", max_votes=600)
    # plot_party_vote_by_digit_relationships(df, "Momentum", max_votes=600)

    plot_party_vote_by_digit_relationships(df, "Fidesz", max_votes=600,
                                           ref_digit=USE_MEAN_DIGIT_GROUP,
                                           n_bins=60)
    plot_party_vote_by_digit_relationships(df, "DK", max_votes=600,
                                           ref_digit=USE_MEAN_DIGIT_GROUP,
                                           n_bins=60)
    plot_party_vote_by_digit_relationships(df, "Momentum", max_votes=600,
                                           ref_digit=USE_MEAN_DIGIT_GROUP,
                                           n_bins=60)

    # plot_party_vote_by_digit_relationships(df, "Fidesz", max_votes=600, ref_digit=1)
    # plot_party_vote_by_digit_relationships(df, "DK", max_votes=600, ref_digit=1)
    # plot_party_vote_by_digit_relationships(df, "Fidesz", max_votes=600, ref_digit=0)
    # plot_party_vote_by_digit_relationships(df, "DK", max_votes=600, ref_digit=0)
    # plot_party_vote_by_digit_relationships(df, "Ervenyes", max_votes=600, ref_digit=0)
    # plot_party_vote_by_digit_relationships(df, "Nevjegyzekben", max_votes=600, ref_digit=0)
