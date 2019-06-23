from collections import OrderedDict
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from digit_entropy_distribution import prob_of_entr, get_entropy


FIDESZ_NO = 13

df = pd.read_csv("2018/parties_20180415.csv", delimiter=";")
df["ld"] = df.Votes.apply(lambda x: x % 10)

def get_settlement(x):
    return " ".join(x.split(" ")[:-1])

df["Settlement"] = df.Ward.apply(get_settlement)
df["Ward_No"] = df.Ward.apply(lambda x: x.split(" ")[-1])

numbers = df.loc[(df.Division=="HEVES 01.") &
                 (df.Party_No==FIDESZ_NO) &
                 (df.Ward.apply(lambda x: x.startswith("Eger")))]

plt.hist(numbers.ld)
plt.show()


# overall Heves 01
numbers = df.loc[(df.Division=="HEVES 01.") &
                 (df.Party_No==FIDESZ_NO)]

plt.hist(numbers.ld)
plt.show()


# overall Heves 02
numbers = df.loc[(df.Division=="HEVES 02.") &
                 (df.Party_No==FIDESZ_NO)]

plt.hist(numbers.ld)
plt.show()

# overall Fidesz
numbers = df.loc[df.Party_No==FIDESZ_NO]

plt.hist(numbers.ld)
plt.show()


numbers = df.loc[(df.Division=="BUDAPEST 01.") &
                 df.Settlement.str.startswith("Budapest I.") &
                 (df.Party_No==FIDESZ_NO)]

plt.hist(numbers.ld)
plt.show()

# overall "big" (say "city") Fidesz
numbers = df.loc[(df.Party_No==FIDESZ_NO) &
                 (df.Votes >= 200)]

plt.hist(numbers.ld)
plt.show()


"""
Question: what are the suspect5 cities in
case of this data set?

The 2019 one is organised by settlement.

So the two may not be comparable.

Is that intentional, or is that an obligation?
"""

df_Fidesz = df.loc[df.Party_No == FIDESZ_NO]

cols = OrderedDict([("ld", get_entropy), ("Votes", [min, len])])

# df_Fidesz_entr = df_Fidesz.groupby(["Division", "Settlement"]).aggregate(cols)

# grouping solely by Settlement (apparently results in no collisions)
# in order for comparability with the 2019 data set where certain boundaries
# such as that of Miskolc are different (two divisions vs. a single one)
df_Fidesz_entr = df_Fidesz.groupby(["Settlement"]).aggregate(cols)
df_Fidesz_entr.reset_index(inplace=True)
df_Fidesz_entr.columns = ["Settlement", "ld_entropy", "min", "count"]
df_Fidesz_entr = df_Fidesz_entr.loc[~np.isnan(df_Fidesz_entr.ld_entropy)]
df_Fidesz_entr.sort_values(["ld_entropy"], inplace=True)

def calc_prob(row):
    return prob_of_entr(row["count"], row["ld_entropy"])

df_Fidesz_entr["prob_of_entr"] = \
    df_Fidesz_entr.apply(lambda x: calc_prob(x), axis=1)

df_Fidesz_entr.sort_values(["prob_of_entr"], inplace=True)


df_Fidesz_entr.to_csv("Fidesz_entr_prob_2018.csv", index=False)


# suspects5_digits = pd.merge(suspects5_18, df_Fidesz, how="inner", on=["Division", "Settlement"])
# # this leaves us with only 146 settlements to examine
# suspects5_18 = df_Fidesz_entr.loc[df_Fidesz_entr["min"] >= 201].head(20)
# suspects5_18.to_csv("suspects5_18.csv")

# plt.hist(suspects5_digits.ld, bins=10)
# plt.show()

# print(suspects5_18[suspects5_18.Settlement == "Eger"])
