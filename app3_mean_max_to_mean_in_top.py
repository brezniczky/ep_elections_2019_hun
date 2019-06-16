import numpy as np
from datetime import datetime
from preprocessing import get_cleaned_data
from digit_stat_data import get_last_digit_stats


df = get_cleaned_data()
county_town_digit_sums, digit_sum_extr = get_last_digit_stats(df)


def simulate_mean_max_to_mean_of_top_dist(df_raw, actual_sum_max_to_mean, actual_mean_max_to_mean):
    print()  # prepare for downcount
    hits_sum = 0
    hits_max_to_mean = 0
    df = df_raw.copy()
    start = datetime.now()
    print("N_ITERATIONS", N_ITERATIONS)
    for i in range(N_ITERATIONS):
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
        digit_sum_extr[("ld_Fidesz", "max_to_mean")] = digit_stats["max"] / digit_stats["mean"]

        """
        Note: first 'suspects' were considered too fragile, based on
        max_to_min >= 2 and suspects2["sum"] >= 20
        and have been removed from this script.
        """

        suspects3 = digit_sum_extr.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]
        suspects3 = suspects3.ld_Fidesz.sort_values(["max_to_mean"], ascending=[False])
        suspects3 = suspects3.head(N_TOP_WEIRDEST)

        if actual_sum_max_to_mean < sum(suspects3.max_to_mean):
            hits_sum += 1
        if actual_mean_max_to_mean < np.mean(suspects3.max_to_mean):
            hits_max_to_mean += 1

        if i % 100 == 0:
            try:
                dt = datetime.now() - start
                its_per_sec = (i + 1) / dt.total_seconds()
                print("\r%s elapsed, %.2f iterations per sec.            " %
                      (str(dt), its_per_sec),
                      end="")
            except:
                print("\r(failed to calculate iteratons per sec.)              ", end="")
        # print(sum(suspects3.max_to_mean), np.mean(suspects3.max_to_mean))
    print()  # otblast!

    print(hits_sum / N_ITERATIONS * 100)
    print(hits_max_to_mean / N_ITERATIONS * 100)
    return hits_sum / N_ITERATIONS, hits_max_to_mean / N_ITERATIONS


N_TOP_WEIRDEST = 300
MIN_VOTERS_THRESHOLD = 100
N_ITERATIONS = 200

suspects3 = digit_sum_extr.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD].copy()
suspects3 = suspects3.ld_Fidesz
suspects3.sort_values(["max_to_mean"], ascending=[False], inplace=True)
suspects3 = suspects3.head(N_TOP_WEIRDEST)

actual_sum_max_to_mean_3 = sum(suspects3.max_to_mean)
actual_mean_max_to_mean_3 = np.mean(suspects3.max_to_mean)
print("actual data gave", actual_sum_max_to_mean_3, actual_mean_max_to_mean_3)

results = []
for SEED in [1234, 1235, 1236]:
    print("simulating with seed", SEED,
          "voters threshold", MIN_VOTERS_THRESHOLD,
          "iterations", N_ITERATIONS,
          ("top %d weirdest (most skewed)" % N_TOP_WEIRDEST))
    np.random.seed(SEED)
    # just the top 100
    result = simulate_mean_max_to_mean_of_top_dist(
      df, actual_sum_max_to_mean_3, actual_mean_max_to_mean_3
    )
    print("got probabilities", result)
    results.append(result)


# voter threshold: 100

# n_iter, top -> seed=1234, 1235, 1236
# 100, 50 -> 21%
# 100, 100 -> 22%
# 5000, 100 -> 18.34%, 19.08%, 18.46%
# 100, 200 -> 17%
# 100, 300 -> 7%
# 200, 300 -> 8.5%, 9%, 7.5%
# 100, 400 -> 11%, 7%, 9%
# 100, 500 -> 11%, 7%, 9%
# 1000, 500 -> 7%, 8.5%, 8%
# 1000, 1000 -> 7%, 8.5%, 8%
# 1000, 2000 -> 7%, 8.5%, 8%
# 2000, 500 -> 6.85%, 7.35%, 7.65%
# 3000, 500 -> 7.2%, 7.4%, 7.2%
# 5000, 500 -> 7.2%, 7.4%, 7.2%
# 5000, 1000 -> 7.02%, 7.24%, 6.48%


# suspects2 = digit_sum_extr.loc[digit_sum_extr.ld_Fidesz.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
# # suspects2 = suspects2.loc[suspects2.ld_Fidesz["sum"] >= WARD_THRESHOLD]



"""
Check back on these:

MIN_VOTERS_THRESHOLD = 100
MAX_TO_MEAN_THRESHOLD = 1.5
"""
