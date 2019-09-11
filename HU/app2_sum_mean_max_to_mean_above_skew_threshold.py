# Dropped approach

# -*- coding: utf-8 -*-

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import scipy.special
# from HU.preprocessing import get_preprocessed_data


"""
Tried:
MMT; WT
2.0; 20  -> hits: 11/20, ~ 6 %
1.8; 19  -> hits: 15/33, ~ 35 %
1.5; 20  -> hits: 33/63, ~ 1 %
1.5; 15  -> hits: 61/107 ~ 0.01 %
1.5; 10  -> hits: 68/120 ~ 0.01 % or less
1.4; 20  -> hits: 33/71  ~ 5.3 %
1.3; 20  -> hits: 33/72  ~ 6.5 %
"""
import numpy as np
from HU.preprocessing import get_preprocessed_data
from HU.digit_stat_data import (
    get_suspects2, get_last_digit_stats,
    MAX_TO_MEAN_THRESHOLD, MIN_VOTERS_THRESHOLD
)


df = get_preprocessed_data()
_, digit_sum_extr = get_last_digit_stats(df)
suspects2 = get_suspects2(digit_sum_extr)


actual_sum_max_to_mean = sum(suspects2.max_to_mean)
actual_mean_max_to_mean = np.mean(suspects2.max_to_mean)
print(actual_sum_max_to_mean, actual_mean_max_to_mean)

np.random.seed(1235)

N_ITERATIONS = 1000

def simulate_again(df_raw, actual_sum_max_to_mean, actual_mean_max_to_mean):
    hits_sum = 0
    hits_max_to_mean = 0
    for i in range(N_ITERATIONS):
        df = df_raw.copy()
        df.ld_Fidesz = np.random.choice(range(10), len(df))

        county_town_digit_groups = (
            df[["Megye", "Telepules", "ld_Fidesz", "Fidesz"]]
            .groupby(["Megye", "Telepules", "ld_Fidesz"])
        )
        county_town_digit_sums = county_town_digit_groups.aggregate({"ld_Fidesz": len, "Fidesz": min})

        digit_sum_extr = \
            county_town_digit_sums.groupby(["Megye", "Telepules"]) \
            .aggregate({"ld_Fidesz": [max, 'mean'], #  , min, sum, lucky_nr, lucky_nr2],
                        "Fidesz": min})

        digit_stats = digit_sum_extr.ld_Fidesz
        digit_sum_extr["ld_Fidesz", "max_to_mean"] = digit_stats["max"] / digit_stats["mean"]

        """
        Note: first 'suspects' were considered too fragile, based on
        max_to_min >= 2 and suspects2["sum"] >= 20
        and have been removed from this script.
        """

        suspects2 = digit_sum_extr.loc[digit_sum_extr.ld_Fidesz.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
        # suspects2 = suspects2.loc[suspects2.ld_Fidesz["sum"] >= WARD_THRESHOLD]
        suspects2 = suspects2.loc[suspects2.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]

        suspects2 = suspects2["ld_Fidesz"]

        if actual_sum_max_to_mean < sum(suspects2.max_to_mean):
            hits_sum += 1
        if actual_mean_max_to_mean < np.mean(suspects2.max_to_mean):
            hits_max_to_mean += 1
        print(sum(suspects2.max_to_mean), np.mean(suspects2.max_to_mean))
    print(hits_sum / N_ITERATIONS * 100)
    print(hits_max_to_mean / N_ITERATIONS * 100)

simulate_again(df, actual_sum_max_to_mean, actual_mean_max_to_mean)

# gives 3.7%, 3.5% for the actual_mean_max_to_mean value
# the sum would be fine (45% p-value)
# so what?
