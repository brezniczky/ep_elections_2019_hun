from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import List, Any
from drdigit.digit_entropy_distribution import (
    prob_of_entr, get_entropy, prob_of_twins,
    _DEFAULT_PE_RANDOM_SEED, _DEFAULT_PE_ITERATIONS
)
from drdigit.digit_correlations import equality_prob_vector


def get_overhearing_scores(group_ids,
                           base_columns, indep_columns):  # , seed, iterations):

    indep_colnames = ["col_%d" % i for i in range(len(indep_columns))]
    indep_dict = {
        indep_colname: col
        for col, indep_colname in zip(indep_columns, indep_colnames)
    }

    dfs = [
        pd.DataFrame(dict(base=base, group_id=group_ids, **indep_dict))
        for base in base_columns
    ]

    def equality_prob_coeff_vector_df(df):
        return pd.Series(equality_prob_vector(df["base"],
                                              [df[colname]
                                              for colname in indep_colnames]))

    def geom_mean(x):
        return np.prod(x) ** (1 / len(x))

    coeff_dfs = []
    for df in dfs:
        coeff_df = df.groupby(["group_id"]).apply(equality_prob_coeff_vector_df)
        coeff_dfs.append(coeff_df)

    concatenated_df = pd.concat(coeff_dfs, axis=1)

    return concatenated_df.apply(geom_mean, axis=1)


def get_group_scores(group_ids: List[Any], digits: List[int],
                     overhearing_base_columns: List[List[int]]=[],
                     overhearing_indep_columns: List[List[int]]=[],
                     seed: int=_DEFAULT_PE_RANDOM_SEED,
                     iterations: int=_DEFAULT_PE_ITERATIONS) -> pd.Series:


    # TODO: can add seeds instead of a shared seed
    def agg_prob_of_entr(x: List[int]):
        return prob_of_entr(len(x), get_entropy(x),
                            seed=seed, iterations=iterations)

    df = pd.DataFrame(dict(group_id=group_ids, digit=digits))
    agg = df.groupby(["group_id"]).aggregate([agg_prob_of_entr, prob_of_twins])

    if len(overhearing_base_columns) > 0 and len(overhearing_indep_columns) > 0:
        overhearing_scores = get_overhearing_scores(
            group_ids,
            overhearing_base_columns,
            overhearing_indep_columns,
        )

        agg["overhearing_prob"] = overhearing_scores

    agg = agg.apply(np.prod, axis=1)
    # TOOD: is it really a series?
    return agg


if __name__ == "__main__":
    print(get_group_scores([0, 0, 0, 0,
                            1, 1],
                           [5, 6, 3, 6,
                            5, 5]))
    print(get_overhearing_scores(
        group_ids=[1, 1, 2, 2],
        base_columns=[[1, 2, 3, 4]],
        indep_columns=[[2, 3, 3, 4], [10, 11, 11, 10]],
    ))
