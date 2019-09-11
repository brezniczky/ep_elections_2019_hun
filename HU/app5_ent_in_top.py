"""
(Approach included in the notebook currently, but excluded
from the generic processing sequence.)

Here particular electoral settlements will be focused on.

Only those are of interest where
- the Fidesz party got at least 100 votes - this suggests that the last digit
  distribution should be close to uniform, which the simulation relies on
- there are at least 10 electoral wards in the settlement - so that entropy
  estimates are more accurate, the need to level them is less relevant

When humans generate random numbers or tamper with values, the entropy is
expected to reduce as humans tend to avoid/favour certain values.

Given a simulation generating random laydowns of votes (only generating the
last digits), how likely is it to obtain a settlement information entropy total
as low as that of the actual 2019 data?
"""
import numpy as np
from datetime import datetime
from HU.preprocessing import get_preprocessed_data
from HU.digit_stat_data import get_last_digit_stats
from scipy.stats import entropy
import pandas as pd
from typing import List
from arguments import is_quiet, load_output, save_output, is_quick
from drdigit import plot_entropy_distribution


def total_freq_ent(counts):
    # a) the smaller weren't giving results
    # b) the smaller are typically less interesting
    #    (politically), so in this filtering is a "proxy to"
    #    focusing on a segment
    # c) the smaller required a conversion which I
    #    do not trust too much at this point, partly
    #    due to a)
    if len(counts) < 10:
        return np.log(10)
    """
    In [158]: entropy([0.5, 0.5])
    Out[158]: 0.6931471805599453

    In [159]: entropy([0.25, 0.25, 0.25, 0.25])
    Out[159]: 1.3862943611198906

    so at the end I should level these values
    """
    if len(counts) == 1:
        # indefinite-like form -
        # all symbols occur once, so "max"
        # but at the same time, a single symbol
        # occurs only, so all of them are the same -
        # then is it a zero or anything else?
        # my "local" answer: for now it's irrelevant in our
        # analysis as lower values are of interest, so push
        # this to the "big" end (unlikely a top candidate)
        return np.log(10)
    counts = np.array(counts)
    res = entropy(counts)
    if len(counts) < 10:
        # this never executes now
        max_digits_that_appear = min(np.sum(counts), 10)
        if max_digits_that_appear != 10:
            res = res * np.log(10) / np.log(max_digits_that_appear)
    return res


def get_simulated_total_ent_prob(df_raw, actual_total_ent, iterations):
    print()  # prepare for downcount
    hits = 0
    df = df_raw.copy()
    start = datetime.now()
    entropies = []
    for i in range(iterations):
        df.ld_Fidesz = np.random.choice(range(10), len(df))

        county_town_digit_groups = (
            df[["Megye", "Telepules", "ld_Fidesz", "Fidesz"]]
            .groupby(["Megye", "Telepules", "ld_Fidesz"])
        )
        county_town_digit_sums = county_town_digit_groups.aggregate(
            {"ld_Fidesz": len, "Fidesz": min}
        )
        county_town_ent = (county_town_digit_sums
                           .groupby(["Megye", "Telepules"])
                           .aggregate({"ld_Fidesz": total_freq_ent,
                                       "Fidesz": min}))
        suspects5 = county_town_ent.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]
        suspects5 = suspects5.sort_values(["ld_Fidesz"], ascending=[True])
        suspects5 = suspects5.head(N_TOP_WEIRDEST)

        sim_total_ent = np.sum(suspects5.ld_Fidesz)

        if sim_total_ent <= actual_total_ent:
            hits += 1
        entropies.append(sim_total_ent)

        if i % 20 == 0:
            try:
                dt = datetime.now() - start
                its_per_sec = (i + 1) / dt.total_seconds()
                print("\r%s elapsed, %d/%d done, %.2f iterations per sec.            " %
                      (str(dt), i + 1, iterations, its_per_sec),
                      end="")
            except:
                print("\r(failed to calculate iteratons per sec.)              ", end="")
    print()  # otblast!
    print(hits / iterations * 100)

    return hits / iterations, entropies


def save_results(actual_total_ent, probabilities, entropies):
    df = pd.DataFrame(dict(names=["actual_total_ent"],
                           values=[actual_total_ent]))
    save_output(df, "app5_actual_total_ent.csv")

    df = pd.DataFrame(dict(probability=probabilities))
    save_output(df, "app5_ent_in_top_probs.csv")

    df = pd.DataFrame(dict(entropy=entropies))
    save_output(df, "app5_ent_in_top_entropies.csv")


def load_results():
    df = load_output("app5_actual_total_ent.csv")
    actual_total_ent = df["values"][0]

    df = load_output("app5_ent_in_top_probs.csv")
    probabilities = df.probability

    df = load_output("app5_ent_in_top_entropies.csv")
    entropies = df.entropy

    return actual_total_ent, probabilities, entropies


def plot_app5_entropy_distribution():
    actual_total_ent, probabilities, entropies = load_results()
    plot_entropy_distribution(actual_total_ent, np.mean(probabilities),
                              entropies, is_quiet=is_quiet())


if __name__ == "__main__":
    if not "df" in globals():
        df = get_preprocessed_data()
    if not "county_town_digit_sums" in globals():
        county_town_digit_sums, digit_sum_extr = get_last_digit_stats(df)

    N_TOP_WEIRDEST = 20
    MIN_VOTERS_THRESHOLD = 100
    N_ITERATIONS = 1000 if not is_quick() else 5
    SEEDS = [1234, 1235, 1236, 1237]

    county_town_ent = (county_town_digit_sums
                       .groupby(["Megye", "Telepules"])
                       .aggregate({"ld_Fidesz": total_freq_ent,
                                   "Fidesz": min}))
    suspects5 = county_town_ent.loc[digit_sum_extr.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]
    suspects5 = suspects5.sort_values(["ld_Fidesz"], ascending=[True])
    suspects5 = suspects5.head(N_TOP_WEIRDEST)

    actual_total_ent = np.sum(suspects5.ld_Fidesz)

    print("actual data gave", actual_total_ent)


    probabilities = []
    all_entropies = []
    for seed in SEEDS:
        print("simulating with seed=", seed,
              "voters threshold=", MIN_VOTERS_THRESHOLD,
              "iterations=", N_ITERATIONS,
              ("; top %d weirdest (most skewed)" % N_TOP_WEIRDEST))
        np.random.seed(seed)
        # just the top 100
        probability, entropies = get_simulated_total_ent_prob(
            df, actual_total_ent, N_ITERATIONS
        )
        probabilities.append(probability)
        all_entropies.extend(entropies)


    print("Simulations were carried out with seeds",
          ",".join([str(seed) for seed in SEEDS])),
    print("Resulting probabilities of such a low overall entropy:",
          ", ".join(["%.2f %%" % (p * 100) for p in probabilities]))
    print("Mean probability: %.2f %%" % (100 * np.mean(probabilities)))

    save_results(actual_total_ent, probabilities, all_entropies)

    plot_app5_entropy_distribution()


"""
Wisdom of the day:
1) the common definition of entropy is infeasible for short sequences
the emp. frequencies (for two symbols)
(2 1)
can both suggest that this is a uniform or a 2:1 distribution
(or something else, for the matter, primarily in the range)

However it is treated as a definite thing.

If I were to take each configuration as a symbol, generate
and assign a probability to each, that would mean I could
calculate a joint probability of all of them appearing
together.

Anyway.
The top 20 suspects giving a difficult to "cherry-pick"
from random data (3% probability of happening by chance)
entropy:

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
