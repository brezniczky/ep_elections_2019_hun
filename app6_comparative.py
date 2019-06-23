from preprocessing import get_cleaned_data
# from digit_stat_data import get_last_digit_stats
from digit_entropy_distribution import get_entropy, prob_of_entr
from containers import OrderedDict


# import numpy as np
# from datetime import datetime
# from digit_stat_data import get_last_digit_stats
# from scipy.stats import entropy
# import pandas as pd


if not "df" in globals():
    df = get_cleaned_data()

# if not "county_town_digit_sums" in globals():
#     county_town_digit_sums, digit_sum_extr = get_last_digit_stats(df)


agg_cols = OrderedDict([("ld_Fidesz", [get_entropy, len])])


df_Fidesz_ent = df.groupby(["Telepules"]).aggregate(agg_cols)
df_Fidesz_ent.reset_index(inplace=True)
df_Fidesz_ent.columns = ["Settlement", "ld_entropy", "count"]


def calc_prob(row):
    return prob_of_entr(row["count"], row["ld_entropy"])

df_Fidesz_ent["count"] = df_Fidesz_ent["count"].astype(int)
df_Fidesz_ent["prob_of_entr"] = df_Fidesz_ent.apply(calc_prob, axis=1)

df_Fidesz_ent.sort_values(["prob_of_entr"], inplace=True)

df_Fidesz_ent_2018 = pd.read_csv("Fidesz_entr_prob_2018.csv")

df_comparative = pd.merge(df_Fidesz_ent, df_Fidesz_ent_2018, how="inner", on=["Settlement"])
df_comparative["prob_of_both"] = df_comparative.prob_of_entr_x * df_comparative.prob_of_entr_y
df_comparative.sort_values(["prob_of_both"], inplace=True)

suspects = df_comparative.loc[(df_comparative.prob_of_both < 0.05) & (suspects["min"] >= 100 )]


sum(suspects["min"] * suspects.count_y)

# 166797 votes are potentially affected by the irregularities
# in the above suspects

suspects.columns = ["Settlement", "ld_Entropy_2019", "count_2019", "p_2019", "ld_entropy_2018", "min_votes_2018", "count_2018", "p_2018", "p_combined"]
suspects.to_csv("comparative_result_suspects.csv", index=False)

