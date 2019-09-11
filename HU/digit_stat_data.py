import numpy as np


def lucky_nr(occurrances):
    digits = np.array([x[-1]
                       for x in
                       list(occurrances.index)])
    counts = np.array(list(occurrances))
    tops = digits[counts==max(counts)]
    if len(tops) == 1:
        return tops[0]
        # return np.random.choice(tops, 1)[0]
    else:
        return None


def lucky_nr2(occurrances):
    digits = np.array([x[-1]
                       for x in
                       list(occurrances.index)])
    counts = np.array(list(occurrances))
    tops = digits[counts==max(counts)]
    if len(tops) == 1:
        if len(counts) > 1:
            top2s = digits[counts==(sorted(counts)[-2])]
            if len(top2s) == 1:
                return top2s[0]
            # return np.random.choice(top2s, 1)[0]
    else:
        return None

def get_last_digit_stats(df):
    county_town_digit_groups = (
        df[["Megye", "Telepules", "ld_Fidesz", "Fidesz"]]
        .groupby(["Megye", "Telepules", "ld_Fidesz"])
    )
    county_town_digit_sums = county_town_digit_groups.aggregate({"ld_Fidesz": len, "Fidesz": min})


    digit_sum_extr = \
        county_town_digit_sums.groupby(["Megye", "Telepules"]) \
        .aggregate({"ld_Fidesz": [max, 'mean', min, sum, lucky_nr, lucky_nr2],
                    "Fidesz": min})


    digit_stats = digit_sum_extr.ld_Fidesz
    digit_sum_extr["ld_Fidesz", "max_to_min"] = digit_stats["max"] / digit_stats["min"]
    digit_sum_extr["ld_Fidesz", "max_to_mean"] = digit_stats["max"] / digit_stats["mean"]
    digit_sum_extr["ld_Fidesz", "max_less_min_to_mean"] = \
        (digit_stats["max"] - digit_stats["min"]) / digit_stats["mean"]

    return county_town_digit_sums, digit_sum_extr


MAX_TO_MEAN_THRESHOLD = 2.0
# WARD_THRESHOLD = 20
MIN_VOTERS_THRESHOLD = 100


def get_suspects2(digit_sum_extr):
    suspects2 = digit_sum_extr.loc[digit_sum_extr.ld_Fidesz.max_to_mean >= MAX_TO_MEAN_THRESHOLD]
    # suspects2 = suspects2.loc[suspects2.ld_Fidesz["sum"] >= WARD_THRESHOLD]
    suspects2 = suspects2.loc[suspects2.Fidesz["min"] >= MIN_VOTERS_THRESHOLD]

    suspects2 = suspects2["ld_Fidesz"]

    return suspects2
