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


suspects = digit_sum_extr.loc[(digit_sum_extr.max_to_min >= 2)]

# it's also the number of electoral wards
suspects = suspects.loc[suspects["sum"] >= 20]

print("suspicious towns")
print(suspects)
print("total: %d suspicious towns" % len(suspects))

suspects2 = digit_sum_extr.loc[digit_sum_extr.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
suspects2 = suspects2.loc[suspects2["sum"] >= 20]


print("suspicious towns based on max to mean digit occurrence ratio")

suspects2.sort_values(["max_to_mean"], ascending=[False], inplace=True)
print(suspects2)
suspects2.to_csv("suspects2.csv", encoding="utf8")

suspects2.reset_index()

suspects2.sort_values(["Megye", "lucky_nr", "Telepules"], ascending=[False, True, True], inplace=True)
s2_alt_sort = suspects2.reset_index()[["Megye", "Telepules", "min_lucky", "lucky_nr", "lucky_nr2"]].sort_values(["Megye", "Telepules"])
s2_alt_sort.to_csv("suspects2_alt_sort.csv", encoding="utf8")
# max-min ratio versus sum plot

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
# suspects2.groupby(["lucky_nr"]).aggregate(len)

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

# assume independent draws of 63 digits, where none of them is 7
# thus there are 9 options
# this gives 9 ^ 63
# then divide by 10 ^ 63, the total number of unrestricted options

P = 0.9 ** len(suspects2)

print("probability = %f %%" % (P * 100))
print("this is a rough estimate of the probability that missing 7's\n"
      "is just the matter of chance")
print("in other words, a %.2f %% chance of a manipulation" % ((1 - P) * 100))


print("\n\nlet's try to model the digit distribution now more accurately")
P7 = county_town_digit_sums.groupby(["ld_Fidesz"]).aggregate(sum)[7.0] / sum(county_town_digit_sums)
P = (1 - P7) ** len(suspects2)

print("probability = %f %%" % (P * 100))
print("this is an improved estimate of the probability that missing 7's\n"
      "is just the matter of pure chance")
print("in other words, a %.2f %% chance of a manipulation" % ((1 - P) * 100))


print("\n\nlet's check the probability of any digit disappearing\n"
      "from the distibution of those in the suspicious areas")
pop_last_digit_freqs = county_town_digit_sums.groupby(["ld_Fidesz"]).aggregate(sum)
Pi = [pop_last_digit_freqs[float(i)] / sum(pop_last_digit_freqs) for i in range(10)]
P = sum((1 - Pi[i]) ** len(suspects2) for i in range(10))

print("probability = %f %%" % (P * 100))
print("this is a super-improved estimate of the probability that any of the\n"
      "digits missing from the top suspects' last digits is just the\n"
      "matter of pure chance")
print("in other words, a %.2f %% chance of a manipulation" % ((1 - P) * 100))

print("actually, this might be an underestimation at this point, as the\b"
      "other digits also deviate from their 'normal' frequencies - which is\n"
      "completely unaccounted for. however, picking the most distorted areas\n"
      "should probably increase such deviations on the other hand.")

# TODO: plot max/mean ratio by area size (voters)

# I know what we need!
# a likelihood estimate
# well, and what about the last digit?
# the count of the last digit is always n-(all others)

# now if I only consider the
# first 9

# pick 63 digits from 10 possibilities
# what are the odds that none of them is the #7th?
# what are the odds that one of them is missing?

"""
Okay, let's try some Monte-Carlo-ish approach
"""

last_digit_pop = [int(x) for x in df.ld_Fidesz if not np.isnan(x)]

np.random.seed(1234)

# takes a couple of minutes, gives about 1.32% in 100k iterations,

n = 0
for i in range(100000):
    s = np.random.choice(last_digit_pop, len(suspects2))
    all_present = len(set(s)) == 10
    if all_present:
        n += 1
    if i % 1000 == 0:
        print(i, n, "%.2f %%" % ((i + 1 - n) / (i + 1) * 100))

print(i, n, "%.2f %%" % ((i + 1 - n) / (i + 1) * 100))

prob_from_general = (i + 1 - n) / (i + 1)

print("whilst the above were estimations with possibly missing\n"
      "statements of assumptions, given a simulation of \n"
      "draws with replacement, the chance of the draw of the %d\n"
      "weirdest areas' last digits being from the 'general'\n"
      "sample is still found to be about %.2f %%" %
      (len(suspects2), prob_from_general * 100))
print("\nthis - if correct - still renders the numbers very\n"
      "suspicious. []\n")
print("even more so, since only the information that one of the\n"
      "digits is completely missing from the sample was leveraged\n"
      "however there are other frequency deviations from the mean\n"
      "there as well, which should (I believe) further reduce\n"
      "the odds of this configuration of occurrences appearing.")


# and then there's the potentially biasing effect of
# sampling for skewed frequencies - how do you adjust for
# that? simple - try to apply the full data processing to
# the resampled data and see how that affects the occurrence
# of completely missing digits in the data
#
# let's try to go for it (may get too slow though ...)
def test_full_process_with_remodelled_data(df_to_copy):
    df = df_to_copy.copy()
    # there's a NaN-ful row, exclude that
    df = df.loc[df.Megye.apply(type) == str]

    # gave a 4.0 percent chance - with 1000 iterations
    # (TODO: despite the seed I may have seen inconsistent results?)
    np.random.seed(4321)

    def as_index(values):
        # ensure it is ordered for reproducibility
        # (sets have a slightly random order)
        value_set_list = sorted(list(set(values)))
        value_index_dict = dict(
          zip(list(value_set_list), range(len(value_set_list)))
        )
        return [value_index_dict[v] for v in values]

    # combining the typical aggregation key apparently gets
    # things slightly faster
    df["Megyepules"] = df.apply( \
      lambda x: x.Telepules + "|" + x.Megye,
      axis=1
    )
    df.Megyepules = as_index(df.Megyepules)

    hits = 0
    bulls = 0
    misses = 0
    start = datetime.now()

    for i in range(1000):
        df.ld_Fidesz = (
            np.random.choice(last_digit_pop,
                             len(df))
        )

        # repeat the full process
        town_groups = \
            df[
                ["Megyepules", "ld_Fidesz"]
            ].groupby(["Megyepules", "ld_Fidesz"])

        county_town_digit_sums = town_groups.aggregate(len)

        # max_to_min, min are not used - skip them for speed
        digit_sum_extr = county_town_digit_sums \
                         .groupby(["Megyepules"]) \
                         .aggregate([max, 'mean',
                                     # min,
                                     sum, lucky_nr])
        # digit_sum_extr["max_to_min"] = digit_sum_extr["max"] / digit_sum_extr["min"]
        digit_sum_extr["max_to_mean"] = digit_sum_extr["max"] / digit_sum_extr["mean"]

        # seems more selective than the next
        suspects2 = digit_sum_extr.loc[digit_sum_extr.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
        suspects2 = suspects2.loc[suspects2["sum"] >= 20]

        if len(set(suspects2.lucky_nr)) == 10:
            misses += 1
        else:
            hits += 1
            digits_and_counts = \
              np.unique(suspects2.lucky_nr, return_counts=True)
            sorted_counts = sorted(digits_and_counts[1], reverse=True)
            if (sorted_counts[0] +
                sorted_counts[1] +
                sorted_counts[2] >= 27 / 63 * sum(sorted_counts)):

                    bulls += 1

        if i % 5 == 0:
            print(datetime.now() - start, i, hits, misses, bulls)

    print("hits:", hits, "bull hits:", bulls, "misses:", misses)
    return hits / (hits + misses), bulls / (hits + misses)


if input("Run full (slow) remodelling style simulation? (Y/N)").lower() ==  "y":
  P_mc, P_bull_mc = test_full_process_with_remodelled_data(df)
  print("percentage of hits (i.e. extreme digit \n"
        "draws with at least one digit missing using\n"
        "resampled actual data): %.2f %%" %
        (P_mc * 100))

  print("percentage of 'bull' hits (i.e. extreme digit \n"
        "draws with at least one digit missing and three of\n"
        "them occurring at least 27/63 of the cases using\n"
        "resampled actual data): %.2f %%" %
        (P_bull_mc * 100))  # hello Pitbull MC!


"""
In fact the distribution is near-uniform for these suspects,
so the above (costly) simulation is a peace of mind thing.
"""


"""
We can - why not - instead examine something else:
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
"natural" quasi-geographic ordering reveals a very potential
location-dependent relationship in terms of last digits, in
case of the most suspiciously looking settlements.

In case of these areas chosen by size (number of electoral
wards >= 20) and sufficiently skewed last digit distribution
(most frequent is at least 1.5x as frequent as its
expected frequency, the latter being number of electoral wards / 10),
we find that the top two most frequent digits in these areas
tend to correlate with those in the area that follows,
more often than implied by pure chance.
"""
)


def test_digit_geographic_correlation_stat(suspects):
    commons = []

    for i in range(len(suspects) - 1):
        r1 = suspects.iloc[i]
        r2 = suspects.iloc[i + 1]
        top_digits1 = set([r1.lucky_nr, r1.lucky_nr2])
        top_digits2 = set([r2.lucky_nr, r2.lucky_nr2])
        common = top_digits1.intersection(top_digits2)
        commons += list(common)

    unique, counts = np.unique(commons, return_counts=True)

    print(list(zip(unique, counts)))
    print("Total number of digits in the intersections:", len(commons))

    missing_digit_count = 10 - len(unique)
    very_rare_digit_count = sum(counts == 1)
    print("Missing digits:", missing_digit_count)
    print("Digits occurring only once:", very_rare_digit_count)
    print("Probability of this occurring randomly from %d "
          "draws of 10 uniformly distributed digits:" % len(commons))

    nd0 = missing_digit_count
    nd1 = very_rare_digit_count
    n_draws = len(commons)

    # so (7 / 10) ** 31 * 32 / 19 * (10 * 9 * 8 / 3 / 2 / 1) * 3 * 100%

    # P = n_okay / n_all
    # the below conversion prevents possible overflows, Python integers
    # are technically unbounded
    nd0 = int(nd0)
    nd1 = int(nd1)
    n_draws = int(n_draws)

    adv = ((10 - nd0 - nd1) ** (n_draws - nd1) *    # the "usual" digits
         (scipy.special.factorial(nd1)) *
         (scipy.special.binom(n_draws, nd1)))
    total = (10 ** n_draws)

    print("%.4f %%" % (adv / total * 100))

suspects2_unordered = digit_sum_extr.loc[digit_sum_extr.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
suspects2_unordered = suspects2_unordered.loc[suspects2_unordered["sum"] >= 20]

test_digit_geographic_correlation_stat(suspects2_unordered)


"""
Missing digits: 7, 8, there's only one occurrence of a 6.

However, the distribution of "Fidesz"-last digits is very
even, especially in case of larger districts.

Note that zeroes are the most frequent above, however by
nature of electoral data, they should come as the least
frequent once zero electoral data rows are removed.

(Unless there are no votes for a party, i.e. the value
is zero, it has to reach up to "9", before they could turn to
"10". It is left to verify but it's almost certain just due
to the sheer size of the above, Fidesz had received votes
in every ward. However, that it's the zero being the most
frequent last digit, will not be leveraged below.)

The probability of 2 digits missing from a draw of 32 is etc.
(see above)
"""

print("""
What this result underpins (letting alone graver concerns) is
that the quality of the processing of votes is likely far
from adequate, the numbers are unlikely to be accurate, since
they seem to be subject to "human intervention", be it simple
mistakes or intentional adjustments.

The original motivation is to decide whether the flaws in
human random number generation are reflected by the data,
(some irregularities could be spotted by the eye), but
obviously these may be different by people - what we find is
there may be some areal cultural influence on how people
err, forge, whatever.

For instance, there is an unreasonable overrepresentation of
zero digits in the Fidesz party vote counts in certain
suspicious wards. These wards bear significance even in terms
of their effect on the overall outcome simply due to their
size or economic relevance (for instance, 18 of the 23
districts of the capital are a bit odd - coincidence? ;) )

The author is aware of local concerns about the cleanness of
the electoral procedures in at least a few of these regions.

Overall voter confidence in the government could greatly benefit
from better processes and a closer supervision.
""")

# TODO: place BP districts on a votes vs. max_to_mean plot
# such as the below
"""
# Plot this:
# max_to_mean ratio against the number of votes in town
plt.scatter(digit_sum_extr["sum"], digit_sum_extr.max_to_mean, alpha=0.5)
plt.scatter(suspects2["sum"], suspects2.max_to_mean, alpha=0.5)
plt.show()
"""
