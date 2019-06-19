"""
Sum of variance of relative frequency of occurrences of digits per settlement
being tested against this distribution of simulated values assuming a uniform
digit distribution.

This basically aims to tell how much the actual distribution of last digits
is deviating
"""

"""
If only taking variance of the pi, the problem we face is:

Eger: 5,7,6,3,2,1,4,2,5,12

total_freq_var([5,7,6,3,2,1,4,2,5,12])
Out[90]: 0.004169307378904481

total_freq_var([1,2])
Out[91]: 0.074074

Even if I multiply up with n ^ (1/2) for CLT,

In [93]: 0.0041693 * (sum([5,7,6,3,2,1,4,2,5,12]) ** 0.5)
Out[93]: 0.028583280725452073

is still less than
0.1283

So unlikeliness (which I try to add up or rather multiply or what, to get a joint probability)

I will truly end up with a likelihood value

Snells like an information entropy thing

I am trying to say that the information entropy is sub-maximal with a 95% confidence?
Am I?
Markovians? And of what order?

Eger's votes are more surprising than the below made up sequence (completely normal)

"""


import numpy as np
from datetime import datetime
from preprocessing import get_cleaned_data
from digit_stat_data import get_last_digit_stats
from scipy.stats import entropy
import pandas as pd


if not "df" in globals():
    df = get_cleaned_data()

if not "county_town_digit_sums" in globals():
    county_town_digit_sums, digit_sum_extr = get_last_digit_stats(df)


def total_freq_var(x):
    # note: in the typical aggregation use case, this information
    # could be further extracted:
    # digits = np.array(x.index.get_level_values(2))

    counts = np.array(x)
    # remaining_digits = set(range(10)).difference(set(digits))


    # mu = sum(counts) / min(10, len(counts))
    # import ipdb; ipdb.set_trace()

    # musq = (mu ** 2)
    # total = sum((counts - mu) ** 2)
    # n_remaining_digits = 10 - len(counts)
    # total += n_remaining_digits * musq
    sum_counts = sum(counts)
    max_digits = min(sum_counts, 10)
    missing_digits = max_digits - len(counts)
    mu_p = 1 / max_digits
    pi = counts / sum(counts)
    var_pi = np.sum((pi - mu_p) ** 2) + (mu_p ** 2) * missing_digits
    return var_pi / max_digits


def total_freq_ent(x):
    # a) the smaller weren't giving results
    # b) the smaller are typically less interesting
    #    (politically), so in this filtering is a "proxy to"
    #    focusing on a segment
    # c) the smaller required a conversion which I
    #    do not trust too much at this point, partly
    #    due to a)
    if len(x) < 10:
        return np.log(10)
    """
    In [158]: entropy([0.5, 0.5])
    Out[158]: 0.6931471805599453

    In [159]: entropy([0.25, 0.25, 0.25, 0.25])
    Out[159]: 1.3862943611198906

    so at the end I will level these values
    """
    if len(x) == 1:
        # indefinite-like form -
        # all symbols occur once, so "max"
        # but at the same time, a single symbol
        # occurs only - is it a zero or anything else?
        # answer: for now it's irrelevant in our analysis
        # as lower values are of interest, push this to
        # the "big" end
        return np.log(10)
    counts = np.array(x)
    max_digits = min(np.sum(counts), 10)
    res = entropy(counts)
    if max_digits != 10:
        res = res * np.log(10) / np.log(max_digits)
    return res


# print(total_freq_ent([5,7,6,3,2,1,4,2,5,12]))  # Eger
# print(total_freq_ent([5,7,6,3,2,1,4,2,12]))


def simulate_total_var_dist(df_raw, actual_total_var):
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
        county_town_digit_sums = county_town_digit_groups.aggregate(
            {"ld_Fidesz": len, "Fidesz": min}
        )
        county_town_var = (county_town_digit_sums
                           .groupby(["Megye", "Telepules"])
                           .aggregate({"ld_Fidesz": total_freq_ent,
                                       "Fidesz": min}))
        suspects5 = county_town_var.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]
        suspects5 = suspects5.sort_values(["ld_Fidesz"], ascending=[True])
        suspects5 = suspects5.head(N_TOP_WEIRDEST)

        sim_total_var = np.sum(suspects5.ld_Fidesz)

        if actual_total_var < sim_total_var:
            hits += 1

        if i % 20 == 0:
            try:
                dt = datetime.now() - start
                its_per_sec = (i + 1) / dt.total_seconds()
                print("\r%s elapsed, %d/%d done, %.2f iterations per sec.            " %
                      (str(dt), i + 1, N_ITERATIONS, its_per_sec),
                      end="")
            except:
                print("\r(failed to calculate iteratons per sec.)              ", end="")
        # print(sum(suspects3.max_to_mean), np.mean(suspects3.max_to_mean))
    print()  # otblast!

    print(hits / N_ITERATIONS * 100)
    return hits / N_ITERATIONS


# N_TOP_WEIRDEST = 50
N_TOP_WEIRDEST = 20
MIN_VOTERS_THRESHOLD = 100
N_ITERATIONS = 3000
SEEDS = [1234] #  1234, 1236, 1237, 1238]


county_town_var = (county_town_digit_sums
                   .groupby(["Megye", "Telepules"])
                   .aggregate({"ld_Fidesz": total_freq_ent,
                               "Fidesz": min}))
suspects5 = county_town_var.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]
suspects5 = suspects5.sort_values(["ld_Fidesz"], ascending=[True])
suspects5 = suspects5.head(N_TOP_WEIRDEST)

actual_total_var = np.sum(suspects5.ld_Fidesz)

print("actual data gave", actual_total_var)


results = []
for seed in SEEDS:
    print("simulating with seed", seed,
          "voters threshold", MIN_VOTERS_THRESHOLD,
          "iterations", N_ITERATIONS,
          ("top %d weirdest (most skewed)" % N_TOP_WEIRDEST))
    np.random.seed(seed)
    # just the top 100
    result = simulate_total_var_dist(
        df, actual_total_var
    )
    print("got probability", result)
    results.append(result)


print("results:", results)


def get_suspects_last_digits(df: pd.DataFrame, suspects: pd.DataFrame):
    suspects = suspects.reset_index()
    df = df.reset_index()
    joined = pd.merge(suspects[["Megye", "Telepules"]],
                      df[["Megye", "Telepules", "ld_Fidesz"]],
                      how="inner", on=["Megye", "Telepules"])
    return joined.ld_Fidesz


def test_suspects_full_entropy(df: pd.DataFrame, suspects: pd.DataFrame,
                               seed=1234, iterations=1000):
    np.random.seed(seed)
    ld_suspects5 = get_suspects_last_digits(df, suspects)
    actual_entropy = entropy(ld_suspects5.value_counts())
    hits = 0
    for i in range(iterations):
        simulated_series = pd.Series(
            np.random.choice(range(10), len(ld_suspects5))
        )
        simulated_entropy = entropy(simulated_series.value_counts())
        # print("simulated entropy:", simulated_entropy)
        if actual_entropy >= simulated_entropy:
            hits += 1
        if i % 1000 == 999:
            print(i + 1, "iterations done")
    print("Probability of getting such a low entropy is %.2f %%" %
          (hits / iterations * 100, ))
    return hits / iterations

test_suspects_full_entropy(df, suspects5, iterations = 10000, seed=1235)
test_suspects_full_entropy(df, suspects5.iloc[:30], iterations = 10000)
test_suspects_full_entropy(df, suspects5.iloc[:20], iterations = 10000)
test_suspects_full_entropy(df, suspects5.iloc[:25], iterations = 10000)

# actual data gave 457.50160660601597
# simulating with seed 1234 voters threshold 100 iterations 200 top 200 weirdest (most skewed)
# [0.96, 0.95, 0.935, 0.975]

# actual data gave 1148.2771345042297
# simulating with seed 1234 voters threshold 100 iterations 200 top 500 weirdest (most skewed)
#
# [0.96, 0.95, 0.935, 0.975]

# actual data gave 227.24309730661128
# simulating with seed 1234 voters threshold 100 iterations 200 top 100 weirdest (most skewed)
#
# [0.96, 0.95, 0.935, 0.975]

# actual data gave 227.24309730661128
# simulating with seed 1234 voters threshold 100 iterations 1000 top 100 weirdest (most skewed)
# [0.939, 0.955, 0.947, 0.944, 0.934]

# actual data gave 112.11384265690897
# simulating with seed 1234 voters threshold 100 iterations 200 top 50 weirdest (most skewed)
# [0.96, 0.95, 0.935, 0.975, 0.955]

# actual data gave 21.408063409481382
# simulating with seed 1234 voters threshold 100 iterations 200 top 10 weirdest (most skewed)
# [0.935, 0.945, 0.955, 0.985, 0.94]

# 1) I should multiprocess it - okay, reproducibility's with me, who's against?
# 2)


# it's probably replicating the "weird event in simulated data" playbook
# so I've got two choices now: more iterations, more seeds, fewer "top"
# I choose


"""
Wisdom of the day:
1) the common definition of entropy is infeasible for short sequneces
the emp. frequencies (for two symbols)
(2 1)
can both suggest that this is a uniform or a 2:1 distribution
(or something else, for the matter, primarily in the range)

however it is treated as a definite 2:1 probability thing

if I were to take each configuration a symbol, generate
and assign a probability to each, that would mean I could
calculate a joint probability of all of them appearing
together

Why would I calculate an anything though?

JUST give a damn.

Well I could calculate county-wide IE too.




concatenated data frame has 10278 rows
actual data gave 21.408063409481382
simulating with seed 1235 voters threshold 100 iterations 4000 top 10 weirdest (most skewed)

1:01:56.731875 elapsed, 1.07 iterations per sec.
93.65
got probability 0.9365


concatenated data frame has 10278 rows
actual data gave 21.408063409481382
simulating with seed 1234 voters threshold 100 iterations 4000 top 10 weirdest (most skewed)

1:06:55.168365 elapsed, 0.99 iterations per sec.
94.0
got probability 0.94




Top 50, 4000 iterations

concatenated data frame has 10278 rows
actual data gave 112.11384265690897
simulating with seed 1235 voters threshold 100 iterations 4000 top 50 weirdest (most skewed)

1:02:14.988104 elapsed, 1.07 iterations per sec.
94.05
got probability 0.9405

concatenated data frame has 10278 rows
actual data gave 112.11384265690897
simulating with seed 1234 voters threshold 100 iterations 4000 top 50 weirdest (most skewed)

1:06:58.076387 elapsed, 0.99 iterations per sec.
94.15
got probability 0.9415



actual data gave 43.43930621517053
simulating with seed 1235 voters threshold 100 iterations 100 top 20 weirdest (most skewed)

0:01:18.258150 elapsed, 81/100 done, 1.04 iterations per sec.
99.0
got probability 0.99



actual data gave 43.43930621517053
simulating with seed 1235 voters threshold 100 iterations 1000 top 20 weirdest (most skewed)

0:18:40.060501 elapsed, 981/1000 done, 0.88 iterations per sec.
98.0
got probability 0.98


actual data gave 43.43930621517053
simulating with seed 1234 voters threshold 100 iterations 1000 top 20 weirdest (most skewed)

0:20:30.986530 elapsed, 981/1000 done, 0.80 iterations per sec.
97.89999999999999
got probability 0.979
results: [0.979]


actual data gave 43.43930621517053
simulating with seed 1234 voters threshold 100 iterations 3000 top 20 weirdest (most skewed)

0:41:25.566769 elapsed, 2981/3000 done, 1.20 iterations per sec.
97.73333333333333
got probability 0.9773333333333334
results: [0.9773333333333334]


with 3000 iterations - simple and quite significant - that is what we like!
vs. a full blown simulation


those digits are rather suspicious


Next: plot up the likeliness by top N with say 20 iterations
So as to find the sweetspot


The top 20 suspects giving a difficult to beat (3%) entropy

                                         ld_Fidesz  Fidesz
Megye             Telepules
BUDAPEST          Budapest I. kerület      2.059306   231.0
HEVES             Eger                     2.107620   109.0
PEST              Nagykőrös                2.125160   132.0
ZALA              Zalaegerszeg             2.135530   125.0
                  Keszthely                2.150060   145.0
BUDAPEST          Budapest XXII. kerület   2.151744   161.0
GYŐR-MOSON-SOPRON Mosonmagyaróvár          2.163149   103.0
BÉKÉS             Békés                    2.166085   137.0
BUDAPEST          Budapest XVI. kerület    2.168465   175.0
HAJDÚ-BIHAR       Hajdúszoboszló           2.180946   123.0
PEST              Dunakeszi                2.182897   164.0
                  Érd                      2.186214   152.0
BUDAPEST          Budapest XIX. kerület    2.186445   118.0
PEST              Abony                    2.187322   109.0
BUDAPEST          Budapest VI. kerület     2.202646   110.0
BÁCS-KISKUN       Kiskunhalas              2.204954   120.0
BUDAPEST          Budapest III. kerület    2.212523   122.0
                  Budapest XIV. kerület    2.219083   155.0
VAS               Szombathely              2.220608   145.0
BUDAPEST          Budapest II. kerület     2.228551   135.0

"""

