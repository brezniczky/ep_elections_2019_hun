from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import List


def _digit_noise_series(n: int) -> np.ndarray:
    # TODO: hm... random seeds?
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


def get_feasible_groups(df: pd.DataFrame, min_n_wards: int, min_votes: int,
                        value_colname: str="Fidesz",
                        group_colname: str="Telepules",
                        smooth_ld_selectivity: bool=True) -> pd.Series:
    agg = (
        df[[group_colname, value_colname]]
        .groupby([group_colname])
        .aggregate(OrderedDict([(group_colname, len), (value_colname, min)]))
    )
    min_value_colname = "min_%s_votes" % value_colname
    agg.columns = ["n_wards", min_value_colname]
    agg = agg.reset_index()

    vote_mins = agg[min_value_colname]
    if smooth_ld_selectivity:
        vote_mins += _digit_noise_series(len(agg))

    return agg.loc[(agg.n_wards >= min_n_wards) &
                   (vote_mins >= min_votes)][group_colname]


def get_feasible_rows(df: pd.DataFrame, min_value: int,
                      min_value_col_idxs: List[int],
                      smooth_ld_selectivity: bool=True) -> pd.DataFrame:
    """
    Filter row by row for eligible values, but ignore the settlement ward
    counts. Can be useful for taking a look at areas with very few wards.

    :param df: Data frame to filter.
    :param min_value: Rows with with each interesting column containing at least
        this many votes will be returned.
    :param min_value_col_idxs: Interesting columns specified by a list-like of
        indexes.
    :param smooth_ld_selectivity: Whether to use smoothing to avoid the
        potential digit-specificity of the filter value to a degree.
    :return: The filtered data frame.
    """
    if smooth_ld_selectivity:
        min_value = _digit_noise_series(len(df)) + min_value
    is_okay = df.iloc[:, min_value_col_idxs].apply(min, axis=1) >= min_value
    return df[is_okay]


def get_feasible_subseries(arr, min_votes):
    # TODO: here's a one (0.5?) off error, well done ;) it goes -5 to 4 - fix it
    arr = arr[arr >= (min_votes - 5) + np.random.choice(range(10), len(arr))]
    return arr
