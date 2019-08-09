import pandas as pd
import app6_comparative as app6
import app8_prob_of_twin_digits as app8
import app14_digit_correlations_Hun as app14


def get_overall_list_all_3_years():
    # compile the results from
    # - 2019, 2018, 2014 entropy
    # - 2019, 2018, 2014 digit duplication
    # - 2019 overhearing

    (app6_df, _, _, _, _, _) = app6.load_results()
    app6_df["Telepules"] = app6_df["Settlement"]
    app6_df["p"] = app6_df["p_all_3"]
    app6_df = app6_df[["Telepules", "p"]]

    app8_df, _ = app8.load_results()
    app8_df["p"] = app8_df["p_all_regular"]
    app8_df = app8_df[["Telepules", "p"]]

    app14_df = app14.get_top_list()[1]
    # chose not to "power back" - per column pair effects can be interrelated
    # this is probably falls on the conservative side, which may not be that bad
    # (otherwise: theoretically 12, but valid / Fidesz votes are doubly counted)
    app14_df["p"] = app14_df.rel_mean_prob  # ** 11
    app14_df = app14_df[["Telepules", "p"]]

    merged = \
        pd.merge(app6_df, app8_df, "inner", ["Telepules"],
                 suffixes=["_1", "_2"])

    merged = \
        pd.merge(merged, app14_df, "inner", ["Telepules"],
                 suffixes=["", "_3"])
    merged.rename(columns={"p": "p_3"}, inplace=True)

    merged["p"] = merged.p_1 * merged.p_2 * merged.p_3

    merged = merged.sort_values(["p"]).reset_index().drop(columns=["index"])
    return merged


def get_overall_list_last_2_years():
    # compile the results from
    # - 2019, 2018, entropy
    # - 2019, 2018, digit duplication
    # - 2019 overhearing

    (app6_df, _, _, _, _, _) = app6.load_results()
    app6_df["Telepules"] = app6_df["Settlement"]
    app6_df["p"] = app6_df["p_all_2"]
    app6_df = app6_df[["Telepules", "p"]]

    app8_df, _ = app8.load_results()
    app8_df["p"] = app8_df["p_last_2_regular"]
    app8_df = app8_df[["Telepules", "p"]]

    app14_df = app14.get_top_list()[1]
    # chose not to "power back" - per column pair effects can be interrelated
    # this is probably falls on the conservative side, which may not be that bad
    # (otherwise: theoretically 12, but valid / Fidesz votes are doubly counted)
    app14_df["p"] = app14_df.rel_mean_prob  # ** 11
    app14_df = app14_df[["Telepules", "p"]]

    merged = \
        pd.merge(app6_df, app8_df, "inner", ["Telepules"],
                 suffixes=["_1", "_2"])

    merged = \
        pd.merge(merged, app14_df, "inner", ["Telepules"],
                 suffixes=["", "_3"])
    merged.rename(columns={"p": "p_3"}, inplace=True)

    merged["p"] = merged.p_1 * merged.p_2 * merged.p_3

    merged = merged.sort_values(["p"]).reset_index().drop(columns=["index"])
    return merged


def generate_files():
    list1 = get_overall_list_all_3_years()
    print(list1.head(45))
    list2 = get_overall_list_last_2_years()
    print(list2.head(45))
    list2.to_csv("app15_overall_list_last_2_years.csv", index=False)


if __name__ == "__main__":
    generate_files()
