"""
Disclaimer: the author is not really a statistician.
He is aware of potentially making a complete fool out of
myself err.. never mind the syntax :)

However, as long as nobody's apparently dealing with these,
we half-layman might as well try to give a hand.
"""
import pandas as pd
import os.path
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from functools import lru_cache
from drdigit.digit_entropy_distribution import LodigeTest
from drdigit.digit_distribution_charts \
    import plot_party_vote_by_digit_relationships
from drdigit.digit_filtering import get_feasible_subseries


cols = [
    "Komisja otrzymała kart do głosowania",
    "Nie wykorzystano kart do głosowania",
    "Lista nr 3 - KKW KOALICJA EUROPEJSKA PO PSL SLD .N ZIELONI",
    "1 LEWANDOWSKI Janusz Antoni",
    "2 ADAMOWICZ Magdalena",
    "Lista nr 4 - KW PRAWO I SPRAWIEDLIWOŚĆ",
    "1 FOTYGA Anna Elżbieta",
    "2 SELLIN Jarosław Daniel",
]
total_col = "Komisja otrzymała kart do głosowania"
min_vote_count = 100
entropy_window_size = 10


def get_cleaned_data():
    csv_filename = "PL/wyniki_gl_na_kand_po_obwodach_sheet_1.csv"
    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename)
            return df
        except Exception as e:
            print("Error reading saved cleaned data, attempting to recreate ...")
            print("  details:", str(e))
    else:
        print("Cleaning data...")

    xlsx_filename = "PL/wyniki_gl_na_kand_po_obwodach.xlsx"
    xls = pd.ExcelFile(xlsx_filename)
    sheet_name = xls.sheet_names[0]
    print("Reading sheet", sheet_name)
    df = pd.read_excel(xlsx_filename, sheet_name=sheet_name)
    df.to_csv(csv_filename, index=False)   # , encoding="utf-8")
    return df


@lru_cache(maxsize=1)
def get_xlsx(filename):
    return pd.ExcelFile(filename)


def get_big_cleaned_data():
    sheet_list_csv_filename = "PL/wyniki_gl_na_kand_po_obwodach_sheet_list.csv"
    sheet_csv_filename = "PL/wyniki_gl_na_kand_po_obwodach_sheet_%s.csv"

    if os.path.exists(sheet_list_csv_filename):
        try:
            df_sheets = pd.read_csv(sheet_list_csv_filename, encoding="utf-8")
            dfs = []
            for sheet_id in df_sheets.id:
                dfs.append(pd.read_csv(sheet_csv_filename % sheet_id))
            return dfs
        except Exception as e:
            print("Error reading saved cleaned data, attempting to recreate ...")
            print("  details:", str(e))

    print("Cleaning data...")

    xlsx_filename = "PL/wyniki_gl_na_kand_po_obwodach.xlsx"
    xls = get_xlsx(xlsx_filename)

    """ I can just save it as a bunch of CSVs and worry later rapidly
        about the elegant format depending on the scenario """
    dfs = []
    sheet_ids = []
    sheet_names = []
    id = 0
    for sheet_name in xls.sheet_names:
        print("Reading sheet", sheet_name)
        df = pd.read_excel(xlsx_filename, sheet_name=sheet_name)
        id += 1
        df.to_csv(sheet_csv_filename % id, index=False, encoding="utf-8")
        sheet_ids.append(id)
        sheet_names.append(sheet_name)
        dfs.append(df)

    df_sheets = pd.DataFrame(dict(id=sheet_ids, sheet_names=sheet_names))
    df_sheets.to_csv(sheet_list_csv_filename, index=False, encoding="utf-8")

    return dfs


def check_column_code_stats(df, quiet=False):
    codes = df[[df.columns[0]]]
    codes.columns = ["area_code"]
    stats = (
        codes.groupby(["area_code"])
        .aggregate({"area_code": len})
        .rename(columns={"area_code": "area_code_count"})
        .reset_index()
    )
    if not quiet:
        print("code frequencies:")
        for i in range(len(stats)):
            row = stats.iloc[i]
            print("  ", row[0], row[1])
    return stats


def check_digit_doubling(df):
    df = df[cols]

    np.random.seed(1234)

    for col in cols:
        nrs = df[col].values
        nrs = get_feasible_subseries(nrs, 100)
        digs = nrs % 10
        # 2nd two digits are the same
        dbls = sum(((nrs % 100) % 11) == 0)
        print(col)
        print("  nr. of suitable numbers", len(digs))
        dup_cnt = sum(digs[1:] == digs[:-1])
        dup_freq = dup_cnt / (len(digs) - 1)
        print("  digit replication rel. freq.:", dup_freq, )
        p = 1 - st.poisson((len(digs) - 1) / 10).cdf(dup_cnt)
        print("  probability: %.2f %%" % (p * 100))
        print("  two last digits are equal rel. freq.:", dbls / len(digs))
        print()


def get_feasible_areas(df, vote_cols, min_n_wards, min_votes):
    # The min_votes function is effectively unused
    area_col = df.columns[0]
    agg_cols = [(area_col, len)] + [(col, min) for col in vote_cols]
    res_colnames = ["n_wards"] + ["min_" + col for col in vote_cols]
    agg = (
        df[[area_col] + list(vote_cols)]
        .groupby([area_col])
        .aggregate(OrderedDict(agg_cols))
    )
    agg.columns = res_colnames
    agg = agg.reset_index()
    cond = agg.n_wards >= min_n_wards
    for col in vote_cols:
        cond = cond & (agg["min_" + col] >= min_votes)
    return agg.loc[cond][area_col]


def check_likelihoods(df, cols, self_test=False):
    """
    Area based filtering by vote is not feasible here - few areas (if any,
    experience suggests none) have votes only above 100 for multiple lists,
    for instance.
    """
    feasible_areas = get_feasible_areas(df, cols, 0, 0)   # 0, 0
    area_col = df.columns[0]
    df = df.loc[df[area_col].isin(feasible_areas)]

    # hence, the df is filtered by min votes on individual row basis
    def row_min_is_ok(row):
        return min([row[col] for col in cols]) >= 101    #  25
    is_ok = df.apply(row_min_is_ok, axis=1)
    df = df.loc[is_ok].copy()

    if self_test:
        for col in cols:
            df[col] = np.random.choice(list(range(100, 110)), len(df))

    tests = {}
    for col in cols:
        print("Column:", col)
        test = LodigeTest(
            df[col].values % 10,
            df[df.columns[0]].values,
            bottom_n=20,
            pe_iterations=50000,
        )
        print("  actual likelihood:", test.likelihood)
        print("  p-value:", test.p)
        tests[col] = test

    return tests


def plot_test_p_value_hists(tests: dict):
    """ I found these plots pretty unhelpful, not that I'm never wrong ;)
        Anyhow, please welcome the ...
    """
    for col, test in tests.items():
        plt.hist(test.p_values.values(), bins=20)
        plt.title(col)
        plt.show()


def calc_p_values_by_area(tests: dict) -> pd.DataFrame:
    # experimental stuff: likeliness by area code
    prod_p_values = defaultdict(lambda: 1)
    for col, test in tests.items():
        act_p_values = tests[col].p_values
        for group_id, p in act_p_values.items():
            prod_p_values[group_id] *= p

    def fmt(key):
        # forgot to cast to str -- int -> area code padding here
        return "%.6d" % key

    order = sorted([fmt(key) for key in prod_p_values.keys()])
    df = pd.DataFrame(dict(
        area_code=order,
        p_value=[prod_p_values[int(area_code)] for area_code in order],
    ))
    return df


def merge_lista_results(dfs, lista_idxs_to_exclude):
    print("merging \"lista\" columns...")
    merged = []
    lista_cols = set()

    for df in dfs:
        for col in df.columns:
            if col.startswith("Lista nr"):
                lista_cols.add(col)

    def get_lista_col_idx(col):
        return int(col.split(" ")[2])

    lista_cols = sorted(lista_cols, key=get_lista_col_idx)
    lista_cols = [col for col in lista_cols
                  if get_lista_col_idx(col)
                  not in lista_idxs_to_exclude]
    cols_to_keep = ["Kod terytorialny gminy"] + lista_cols

    dfs_to_merge = []
    for df in dfs:
        cols_dict = OrderedDict([
            (col_name,
             df[col_name]
             if col_name in df.columns else None)
            for col_name in cols_to_keep
        ])
        dfs_to_merge.append(pd.DataFrame(cols_dict))
    merged = pd.concat(dfs_to_merge)

    for col in merged.columns:
        if col.startswith("Lista"):
            merged["ld_" + col] = merged[col] % 10

    return merged


if __name__ == "__main__":
    # cols = [
    #     "Lista nr 3 - KKW KOALICJA EUROPEJSKA PO PSL SLD .N ZIELONI",
    #     "1 LEWANDOWSKI Janusz Antoni",
    #     "Lista nr 4 - KW PRAWO I SPRAWIEDLIWOŚĆ",
    #     "1 FOTYGA Anna Elżbieta",
    # ]
    # df = get_cleaned_data()
    # # check_digit_doubling(df)
    # # print("Self test (should provide probabilities far from zero)")
    # # check_likelihoods(df, self_test=True)
    # print("Real test")
    # check_likelihoods(df)
    dfs = get_big_cleaned_data()
    merged = merge_lista_results(
        dfs,
        lista_idxs_to_exclude=[8, 9, 10]
    )
    # it is impractical to check on numerous columns, as this will
    # reduce the data set - a constraint for keeping rows is that
    # each column features a big enough value. now imagine that this
    # is a big "and" of constraints, any of them being too small drops
    # the whole row.
    # this test was unsuccessful at uncovering much, but possibly
    # due to lack of data.

    # check_likelihoods(merged, merged.columns[1:])

    # the primary aim is to deal with the two biggest winners

    if input("run lengthy tests? (y/N)").lower().startswith("y"):
        tests = check_likelihoods(merged, merged.columns[3:5])
        plot_test_p_value_hists(tests)
        p_values = calc_p_values_by_area(tests).sort_values(["p_value"])

        print("Area codes with least unlikely correct results:")
        print(p_values.head(20))

        p_values.to_csv("app11_top_suspects.csv")

        print("--- THE BELOW IS WRONG!!! but could useful as a concept, "
              "though doesn't look very stable. due to the reciprocal. ---")
        print("A rough estimate says these events should have occurred out of")
        print("an est. %d cases - there are %d cases." %
              (int(round(sum(1 / p_values.p_value))), len(p_values)))

    plot_party_vote_by_digit_relationships(
        merged,
        'Lista nr 3 - KKW KOALICJA EUROPEJSKA PO PSL SLD .N ZIELONI',
        max_votes=600,
        n_bins=80,
    )
    plot_party_vote_by_digit_relationships(
        merged,
        'Lista nr 4 - KW PRAWO I SPRAWIEDLIWOŚĆ',
        max_votes=600,
        n_bins = 80,
    )


"""
In [213]: sum(1 / p_values.p_value)
Out[213]: 6133.625219317689

In [214]: len(p_values)
Out[214]: 282

"""
    # .. and what we get is that there are likely "inaccuracies" around
    # lista nr. 4's results.
