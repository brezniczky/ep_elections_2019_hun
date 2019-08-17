from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import List
from drdigit.digit_entropy_distribution import (
    prob_of_entr, get_entropy, prob_of_twins,
    _DEFAULT_PE_RANDOM_SEED, _DEFAULT_PE_ITERATIONS
)


def get_group_scores(group_ids, digits,
                     seed=_DEFAULT_PE_RANDOM_SEED,
                     iterations=_DEFAULT_PE_ITERATIONS):
    # TODO: can add seeds instead of a shared seed

    def agg_prob_of_entr(x: List[int]):
        return prob_of_entr(len(x), get_entropy(x),
                            seed=seed, iterations=iterations)


    df = pd.DataFrame(dict(group_id=group_ids, digit=digits))
    agg = df.groupby(["group_id"]).aggregate([agg_prob_of_entr, prob_of_twins])

    agg = agg.apply(np.prod, axis=1)
    return agg


if __name__ == "__main__":
    print(get_group_scores([0, 0, 0, 0,
                            1, 1],
                           [5, 6, 3, 6,
                            5, 5]))
