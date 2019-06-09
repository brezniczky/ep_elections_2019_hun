# -*- coding: utf-8 -*-

from __future__ import print_function
from pandas import DataFrame, read_csv
# import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt

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

def lucky_nr(occurances):
    digits = np.array([x[-1]
                       for x in
                       list(occurances.index)])
    counts = np.array(list(occurances))
    return digits[counts==max(counts)][0]

digit_sum_extr = county_town_digit_sums.groupby(["Megye", "Telepules"]).aggregate([max, 'mean', min, sum, lucky_nr])
digit_sum_extr["max_to_min"] = digit_sum_extr["max"] / digit_sum_extr["min"]
digit_sum_extr["max_to_mean"] = digit_sum_extr["max"] / digit_sum_extr["mean"]

suspects = digit_sum_extr.loc[(digit_sum_extr.max_to_min >= 2)]
suspects = suspects.loc[suspects["sum"] >= 20]

print("suspicious towns")
print(suspects)
print("total: %d suspicious towns" % len(suspects))

suspects2 = digit_sum_extr.loc[digit_sum_extr.max_to_mean >= 1.5]
suspects2 = suspects2.loc[suspects2["sum"] >= 20]

print("suspicious towns based on max to mean digit occurrence ratio")

suspects2.sort_values(["max_to_mean"], ascending=[False], inplace=True)
print(suspects2)
suspects2.to_csv("suspects2.csv", encoding="utf8")

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
