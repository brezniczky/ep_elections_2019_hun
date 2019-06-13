# -*- coding: utf-8 -*-

from __future__ import print_function
from pandas import DataFrame, read_csv
# import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os.path
import scipy.special


MAX_TO_MEAN_THRESHOLD = 1.5


if not os.path.exists("merged.csv"):
    filename = r'EP_2019_szavaz_k_ri_eredm_ny.xlsx'
    # filename = r'short.xlsx'
    print("file: %s" % filename)
    print("reading sheet names ...")
    xls = pd.ExcelFile(filename)
    print("found %d sheets" % len(xls.sheet_names))

    dfs = []

    for name in xls.sheet_names:
        print("reading", name)
        df = pd.read_excel(filename, sheet_name=name)
        print("read %d rows" % len(df))
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv("merged.csv", index=False)
else:
    df = pd.read_csv("merged.csv")

print("concatenated data frame has %d rows" % len(df))

df.columns = [
    "Unnamed", "Megye", "Telepules", "Szavazokor", "Nevjegyzekben", "Megjelent",
    "Belyegzetlen", "Lebelyegzett", "Elteres megjelentektol", "Ervenytelen", "Ervenyes",
    "MSZP", "MKKP", "Jobbik", "Fidesz", "Momentum", "DK", "Mi Hazank", "Munkaspart", "LMP"
]

columns = df.columns.copy()

for column in columns[-10:]:
    df["ld_" + column] = df[column] % 10

town_groups = df[["Megye", "Telepules", "ld_Fidesz"]].groupby(["Megye", "Telepules", "ld_Fidesz"])
county_town_digit_sums = town_groups.aggregate(len)

digit_sums = county_town_digit_sums.groupby(["ld_Fidesz"]).aggregate(len)

def lucky_nr(occurrances):
    digits = np.array([x[-1]
                       for x in
                       list(occurrances.index)])
    counts = np.array(list(occurrances))
    return digits[counts==max(counts)][0]

def lucky_nr2(occurrances):
    """ 2nd most frequently occurring last digit """
    digits = np.array([x[-1]
                       for x in
                       list(occurrances.index)])
    counts = np.array(list(occurrances))
    return digits[counts==sorted(counts)[1]][0] \
           if len(occurrances) > 1 else None

digit_sum_extr = \
    county_town_digit_sums.groupby(["Megye", "Telepules"]) \
    .aggregate([max, 'mean', min, sum, lucky_nr, lucky_nr2])


def get_min_lucky(df):
    return [a if np.isnan(b) else min(a, b)
            for a, b in zip(df.lucky_nr, df.lucky_nr2)]


digit_sum_extr["max_to_min"] = digit_sum_extr["max"] / digit_sum_extr["min"]
digit_sum_extr["max_to_mean"] = digit_sum_extr["max"] / digit_sum_extr["mean"]
digit_sum_extr["min_lucky"] = get_min_lucky(digit_sum_extr)


"""
Note: first 'suspects' were considered too fragile, based on
max_to_min >= 2 and suspects2["sum"] >= 20
and have been removed from this script.
"""

suspects2 = digit_sum_extr.loc[digit_sum_extr.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
suspects2 = suspects2.loc[suspects2["sum"] >= 20]

print("Suspicious areas based on max to mean digit occurrence ratio")

suspects2.sort_values(["max_to_mean"], ascending=[False], inplace=True)
print(suspects2)
suspects2.to_csv("suspects2.csv", encoding="utf8")

suspects2.reset_index()


"""
We get

                                              max       mean  min  sum  lucky_nr  max_to_min  max_to_mean
Megye                Telepules
BUDAPEST             Budapest I. kerület        6   2.000000    1   20         4    6.000000     3.000000
HEVES                Eger                      12   4.700000    1   47         9   12.000000     2.553191
BUDAPEST             Budapest XXIII. kerület    6   2.400000    1   24         4    6.000000     2.500000
PEST                 Nagykőrös                  5   2.000000    1   20         3    5.000000     2.500000
BUDAPEST             Budapest VII. kerület      9   3.800000    1   38         0    9.000000     2.368421
KOMÁROM-ESZTERGOM    Tata                       7   3.000000    1   24         9    7.000000     2.333333
                     Esztergom                  6   2.700000    1   27         9    6.000000     2.222222
BÉKÉS                Orosháza                   8   3.700000    1   37         2    8.000000     2.162162
ZALA                 Zalaegerszeg              11   5.100000    2   51         9    5.500000     2.156863
TOLNA                Dombóvár                   5   2.444444    1   22         3    5.000000     2.045455
PEST                 Vecsés                     7   3.428571    1   24         2    7.000000     2.041667
ZALA                 Nagykanizsa                9   4.444444    1   40         9    9.000000     2.025000
NÓGRÁD               Salgótarján                8   4.000000    1   40         8    8.000000     2.000000
BORSOD-ABAÚJ-ZEMPLÉN Kazincbarcika              6   3.000000    1   30         6    6.000000     2.000000
HAJDÚ-BIHAR          Hajdúszoboszló             4   2.000000    1   20         0    4.000000     2.000000
BÁCS-KISKUN          Kiskunhalas                6   3.000000    1   30         3    6.000000     2.000000
BUDAPEST             Budapest XVI. kerület     12   6.000000    1   60         5   12.000000     2.000000
PEST                 Szentendre                 5   2.500000    1   20         5    5.000000     2.000000
CSONGRÁD             Szentes                    7   3.500000    1   35         6    7.000000     2.000000
HEVES                Gyöngyös                   6   3.000000    1   30         5    6.000000     2.000000
BÁCS-KISKUN          Baja                       7   3.555556    1   32         8    7.000000     1.968750
BUDAPEST             Budapest II. kerület      14   7.400000    3   74         8    4.666667     1.891892
                     Budapest XXII. kerület     7   3.700000    1   37         1    7.000000     1.891892
                     Budapest IX. kerület      10   5.300000    3   53         8    3.333333     1.886792
                     Budapest XIX. kerület      9   4.800000    2   48         4    4.500000     1.875000
PEST                 Érd                        9   4.800000    1   48         0    9.000000     1.875000
JÁSZ-NAGYKUN-SZOLNOK Szolnok                   13   7.000000    5   70         1    2.600000     1.857143
PEST                 Cegléd                     6   3.300000    1   33         5    6.000000     1.818182
BUDAPEST             Budapest XX. kerület      11   6.100000    3   61         3    3.666667     1.803279
GYŐR-MOSON-SOPRON    Sopron                    10   5.555556    2   50         0    5.000000     1.800000
...                                           ...        ...  ...  ...       ...         ...          ...
BARANYA              Mohács                     4   2.300000    1   23         4    4.000000     1.739130
CSONGRÁD             Makó                       5   2.875000    1   23         2    5.000000     1.739130
VAS                  Szombathely               12   6.900000    3   69         5    4.000000     1.739130
BUDAPEST             Budapest XIV. kerület     13   7.500000    2   75         2    6.500000     1.733333
VESZPRÉM             Ajka                       5   2.900000    1   29         1    5.000000     1.724138
BUDAPEST             Budapest V. kerület        4   2.333333    1   21         3    4.000000     1.714286
JÁSZ-NAGYKUN-SZOLNOK Jászberény                 4   2.333333    1   21         4    4.000000     1.714286
BUDAPEST             Budapest III. kerület     19  11.200000    2  112         9    9.500000     1.696429
PEST                 Vác                        5   3.000000    1   30         6    5.000000     1.666667
                     Dunakeszi                  5   3.000000    1   30         3    5.000000     1.666667
                     Szigetszentmiklós          5   3.000000    1   30         0    5.000000     1.666667
GYŐR-MOSON-SOPRON    Mosonmagyaróvár            5   3.000000    1   30         0    5.000000     1.666667
KOMÁROM-ESZTERGOM    Tatabánya                 10   6.000000    3   60         2    3.333333     1.666667
BÉKÉS                Gyula                      5   3.000000    1   30         4    5.000000     1.666667
BUDAPEST             Budapest VI. kerület       5   3.000000    1   30         3    5.000000     1.666667
                     Budapest XII. kerület      9   5.500000    3   55         5    3.000000     1.636364
BARANYA              Komló                      4   2.444444    1   22         3    4.000000     1.636364
BORSOD-ABAÚJ-ZEMPLÉN Ózd                        6   3.700000    2   37         6    3.000000     1.621622
BÁCS-KISKUN          Kiskunfélegyháza           5   3.100000    2   31         6    2.500000     1.612903
PEST                 Gödöllő                    5   3.111111    2   28         8    2.500000     1.607143
VESZPRÉM             Veszprém                   9   5.600000    2   56         2    4.500000     1.607143
CSONGRÁD             Hódmezővásárhely           8   5.000000    1   50         3    8.000000     1.600000
BUDAPEST             Budapest XVIII. kerület   12   7.600000    4   76         5    3.000000     1.578947
                     Budapest XXI. kerület     11   7.000000    3   70         2    3.666667     1.571429
                     Budapest IV. kerület      11   7.000000    4   70         9    2.750000     1.571429
HAJDÚ-BIHAR          Debrecen                  21  13.700000    8  137         0    2.625000     1.532847
SOMOGY               Kaposvár                   9   5.900000    3   59         2    3.000000     1.525424
BUDAPEST             Budapest VIII. kerület     9   6.000000    3   60         5    3.000000     1.500000
BORSOD-ABAÚJ-ZEMPLÉN Miskolc                   24  16.000000   10  160         2    2.400000     1.500000
SOMOGY               Siófok                     4   2.666667    1   24         1    4.000000     1.500000
"""

# Apparently, in these suspicious towns, people disfavour 7's and 1's
print("lucky number distribution")

plt.hist(digit_sum_extr.lucky_nr, bins=10)
plt.title("\"Lucky number\" (most frequent last digit) distribution\n"
          "just about everywhere - note that there's no 'zero' here.")
plt.show()


plt.hist(suspects2.lucky_nr, bins=10)
plt.title("\"Lucky number\" (most frequent last digit) distribution\n"
          "among large enough towns - in which  a skewed digit\n"
          "distribution raised suspicion of tampering - poor 7. :(")
plt.show()

print("\"lucky number\"s actually featured among top suspicious towns")
print(set(suspects2.lucky_nr))

# print("plotting max to mean ratio histogram")

# plt.hist(digit_sum_extr.max_to_mean, bins=100)
# plt.show()

# plt.hist(suspects2.max_to_mean, bins=100)
# plt.show()

# so in these towns "7" was really disfavoured
print("now let's examine the chance of 7 not being featured at all")
print("(incorrectly assuming that the digit distribution is uniform)")
print("""
A forgiving approach - let's check the probability of any
last digit disappearing from the distibution of those
in the suspicious areas:
""")
pop_last_digit_freqs = county_town_digit_sums.groupby(["ld_Fidesz"]).aggregate(sum)
Pi = [pop_last_digit_freqs[float(i)] / sum(pop_last_digit_freqs) for i in range(10)]
P = sum((1 - Pi[i]) ** len(suspects2) for i in range(10))

print("probability = %f %%" % (P * 100))

print("""
Actually, this might be an underestimation at this point, as the
other digits also deviate from their 'normal' frequencies - which is
completely unaccounted for. However, picking the most distorted areas
should probably increase such deviations on the other hand.
""")

# TODO: shall I re-add anything from what I've just deleted?

"""
Note it then that we may not be aware of the correct
distribution for bigger wards, which are most if not all
of the suspects.

However, we can instead examine something else:
- take the top two last digits
- check them out in "natural" order (close to geographic
proximity)
- the intersections are interesting:
  - how many times is there an intersection (if both are
    the same, count as 2)
  - how many of these appearing are different digits?
    (there are 8 digits appearing)
  - how many of these only appear once?  (gives 1)

  - what is the probability of this assuming an even
    distribution?  (then we get a very small number)
"""

print("""
Taking a look at the top two most frequent digits in the
"natural" * ordering reveals a very potential
location-dependent relationship in terms of last digits,
in case of the most suspiciously looking areas.

* This is how the original data is ordered, in a way,
it reflects geographic proximity to a degree

For these areas picked by size (number of electoral
wards >= 20) and sufficiently skewed last digit distribution
(most frequent is at least 1.5x as frequent as its
expected frequency, the latter being number of electoral wards / 10),
we find that the top two most frequent digits in these areas
tend to correlate with those in the area that follows,
more often than implied by pure chance.

(The concept here was to construct some statistic that
increases by correlation between adjacent or nearby areas,
as well as just generally, correlation.)
""")


def get_overlaps(suspects):
    commons = []

    for i in range(len(suspects) - 1):
        r1 = suspects.iloc[i]
        r2 = suspects.iloc[i + 1]
        top_digits1 = set([r1.lucky_nr, r1.lucky_nr2])
        top_digits2 = set([r2.lucky_nr, r2.lucky_nr2])
        common = top_digits1.intersection(top_digits2)
        commons += list(common)
    return commons


suspects2_unordered = digit_sum_extr.loc[digit_sum_extr.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
suspects2_unordered = suspects2_unordered.loc[suspects2_unordered["sum"] >= 20]

overlaps = get_overlaps(suspects2_unordered)


def count_overlaps_quick(numbers1, numbers2):
    # had to be converted from "zipped object" for slicing
    zipped = list(zip(numbers1, numbers2))

    counts_per_row = [
        len(set(act_row).intersection(set(next_row)))
        for act_row, next_row in zip(zipped[:-1], zipped[1:])
    ]
    return sum(counts_per_row)


def do_reference_draws(get_prob_for, weights=None):
    for seed in [4444, 5555, 6666]:
        probs = []
        counts = []
        np.random.seed(seed)
        for i in range(n_resamples):
            numbers1 = np.random.choice(range(10), len(suspects2), p=weights)
            numbers2 = np.random.choice(range(10), len(suspects2), p=weights)

            # small_df = pd.DataFrame(dict(lucky_nr=numbers1, lucky_nr2=numbers2))
            counts.append(count_overlaps_quick(numbers1, numbers2))
        prob = sum(np.array(counts) >= get_prob_for) / len(counts)
        print("Seed: %d, chance of getting at least %d overlaps between %d number pairs is %.2f %%" %
              (seed, get_prob_for, len(suspects2), prob * 100))

n_resamples = 10000

ref_count = \
    count_overlaps_quick(
        suspects2_unordered.lucky_nr,
        suspects2_unordered.lucky_nr2
    )

print("Further reassurance of that where it's skewed, it's not just randomly skewed....")

print("Reference draws: uniform case")
do_reference_draws(get_prob_for=ref_count)

print("Reference draws: non-uniform case (zeroes are 20% more frequent than other digits)")
do_reference_draws(get_prob_for=ref_count, weights=[12 / 102] + [10 / 102] * 9)

print("Reference draws: non-uniform case (zeroes are 100% more frequent than other digits)")
do_reference_draws(get_prob_for=ref_count, weights=[2 / 11] + [1 / 11] * 9)

print("The chance of not getting anything similar is well above 95% in each case.")

print("""
This result underpins (letting alone graver concerns)
that  the numbers are unlikely to be accurate, thus
the quality of the processing of votes may be inadequate,
since they seem to be subject to "human intervention",
be it simple mistakes or something else.

The original motivation was to decide whether the flaws in
human random number generation are reflected by the data,
(some irregularities seemed to be visible to the eye), but
obviously these may be different by people - what we find is
there may be some areal cultural influence on how people
treat numbers, as well as an unlikely deviation from normal
behaviour in at least a couple of ways.

It is easy to spot that there is an unreasonable
overrepresentation of zero digits in the Fidesz party vote
counts in certain suspicious wards. These wards bear
significance even in terms of their effect on the overall
outcome simply due to their size or economic relevance
(some important districts of the capital are top candidates -
coincidence?)
""")


"""
The author is aware of local concerns about the cleanness of
the electoral procedures in at least a few of these regions.


Overall voter confidence in the government could greatly benefit
from better processes and a closer supervision.
"""

# TODO: place BP districts on a votes vs. max_to_mean plot
# such as the below
"""
# Plot this:
# max_to_mean ratio against the number of votes in town
plt.scatter(digit_sum_extr["sum"], digit_sum_extr.max_to_mean, alpha=0.5)
plt.scatter(suspects2["sum"], suspects2.max_to_mean, alpha=0.5)
plt.show()
"""
