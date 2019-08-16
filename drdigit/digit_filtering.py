from collections import OrderedDict
import numpy as np


def _digit_noise_series(n: int) -> np.ndarray:
    # TODO: not 100% about the return value type hint
    """
    Whenever the PMF of vote values is monotonously decreasing, the aggregated
    digit probability PMF is consistently decreasing too. Then when filtering
    with a minimum, the bottommost value will be overrepresented in the
    resulting sample as well as, consistently, the last digit of the minimum
    value, overrepresented in the sample.
    Adding a noise that uniformly distributes over the last digits with a mean
    zero to either side of the filtering inequality should relief such anomalies.

    :param n: number of noise samples to generate.
    :return:
    """
    return np.random.choice(range(10), n) - 4.5


def get_feasible_settlements(df, min_n_wards, min_fidesz_votes,
                             smooth_ld_selectivity=True):
    agg = (
        df[["Telepules", "Fidesz"]]
        .groupby(["Telepules"])
        .aggregate(OrderedDict([("Telepules", len), ("Fidesz", min)]))
    )
    agg.columns = ["n_wards", "min_fidesz_votes"]
    agg = agg.reset_index()

    vote_mins = agg.min_fidesz_votes
    if smooth_ld_selectivity:
        vote_mins += _digit_noise_series(len(agg))

    return agg.loc[(agg.n_wards >= min_n_wards) &
                   (vote_mins >= min_fidesz_votes)]["Telepules"]


def get_feasible_subseries(arr, min_votes):
    # TODO: here's a one (0.5?) off error, well done ;) it goes -5 to 4 - fix it
    arr = arr[arr >= (min_votes - 5) + np.random.choice(range(10), len(arr))]
    return arr
