import numpy as np
from datetime import datetime
from preprocessing import get_cleaned_data
from digit_stat_data import get_last_digit_stats


df = get_cleaned_data()
county_town_digit_sums, digit_sum_extr = get_last_digit_stats(df)


N_TOP_WEIRDEST = 1000
MIN_VOTERS_THRESHOLD = 100
N_ITERATIONS = 2000
SEEDS = [1234, 1235, 1236, 1237]


def simulate_mean_max_to_mean_of_top_dist(df_raw, actual_mean_max_less_min_to_mean):
    print()  # prepare for downcount
    hits = 0
    df = df_raw.copy()
    start = datetime.now()
    for i in range(N_ITERATIONS):
        df.ld_Fidesz = np.random.choice(range(10), len(df))

        county_town_digit_groups = (
            df[["Megye", "Telepules", "ld_Fidesz", "Fidesz"]]
            .groupby(["Megye", "Telepules", "ld_Fidesz"])
        )
        county_town_digit_sums = county_town_digit_groups.aggregate({"ld_Fidesz": len, "Fidesz": min})

        digit_sum_extr = \
            county_town_digit_sums.groupby(["Megye", "Telepules"]) \
            .aggregate({"ld_Fidesz": [max, min, 'mean'],
                        "Fidesz": min})

        digit_stats = digit_sum_extr.ld_Fidesz
        digit_sum_extr[("ld_Fidesz", "max_less_min_to_mean")] = \
            (digit_stats["max"] - digit_stats["min"]) / digit_stats["mean"]

        suspects4 = digit_sum_extr.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]
        suspects4 = suspects4.ld_Fidesz.sort_values(["max_less_min_to_mean"], ascending=[False])
        suspects4 = suspects4.head(N_TOP_WEIRDEST)

        if actual_mean_max_less_min_to_mean < np.mean(suspects4.max_less_min_to_mean ** 2):
            hits += 1

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

    print(hits / N_ITERATIONS * 100)
    return hits / N_ITERATIONS


suspects4 = digit_sum_extr.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD].copy()
suspects4 = suspects4.ld_Fidesz
suspects4.sort_values(["max_less_min_to_mean"], ascending=[False], inplace=True)
suspects4 = suspects4.head(N_TOP_WEIRDEST)

actual_mean_max_less_min_to_mean = np.mean(suspects4.max_less_min_to_mean ** 2)
print("actual data gave", actual_mean_max_less_min_to_mean)

results = []
for seed in SEEDS:
    print("simulating with seed", seed,
          "voters threshold", MIN_VOTERS_THRESHOLD,
          "iterations", N_ITERATIONS,
          ("top %d weirdest (most skewed)" % N_TOP_WEIRDEST))
    np.random.seed(seed)
    # just the top 100
    result = simulate_mean_max_to_mean_of_top_dist(
        df, actual_mean_max_less_min_to_mean
    )
    print("got probability", result)
    results.append(result)


"""
Okay, so checkin' it in squares (well, a little arbitrary unless we express
that it focuses on big differences and rhymes well with comparing standard
deviations from the mean in case of the digit probabilities or ... whatever?
mean variance is what it relates to)

"mean of the squares of the max_less_min_to_mean ratios"

so then we get 4.2%, 4.8%, 4.75% and 4%.

Which is promising, but I'd have to justify that 'squaring'. And below comes
the link how it will get more accurate and look sooo conventional.
"""

"""
Continue with:
https://www.statlect.com/fundamentals-of-statistics/hypothesis-testing-variance
"""
