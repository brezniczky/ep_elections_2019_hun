from typing import List
import pandas as pd
from functools import lru_cache
import os.path
from collections import OrderedDict


_AREA_CODE_COLNAME = "Kod terytorialny gminy"
_VALID_VOTES_COLNAME = "Liczba kart ważnych"
_VOTERS_ELIGIBLE_TO_VOTE_COLNAME = "Liczba wyborców uprawnionych do głosowania"


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


def get_lista_col_idx(col):
    return int(col.split(" ")[2])


class MergedDataInfo():

    def __init__(self, area_columm, valid_votes_column,
                 nr_of_registered_voters_column, lista_columns):
        self._area_column = area_columm
        self._valid_votes_column = valid_votes_column
        self._nr_of_registered_voters_column = nr_of_registered_voters_column
        self._lista_columns = lista_columns

    def __repr__(self):
        return ("MergedDataInfo(\n"
                "    valid_votes_column: %s,\n "
                "    nr_of_registered_voters_column: %s,\n"
                "    lista_columns: %s\n"
                ")"
                % (self.valid_votes_column, self.nr_of_registered_voters_column,
                   ", ".join(self.lista_columns))
                )

    area_column = \
        property(lambda self: self._area_column)  # type: str

    valid_votes_column = \
        property(lambda self: self._valid_votes_column)  # type: str

    nr_of_registered_voters_column = \
        property(lambda self: self._nr_of_registered_voters_column)  # type: str

    lista_columns = \
        property(lambda self: self._lista_columns)  # type: List[str]

    @lru_cache()
    def get_lista_column(self, index: int) -> str:
        for lista_column in self.lista_columns:
            if get_lista_col_idx(lista_column) == index:
                return lista_column
        raise KeyError("Lista column for index %d is not contained." % index)


"""
TODO: add another, per csv auto-selected column: that of the most popular candidate per lista
"""
def merge_lista_results(dfs,
                        # don't ask :) 7, 8, 9 looked too sparse I think
                        # TODO: but #8 just fails with some error ;)
                        lista_idxs_to_exclude=[8],
                        return_overview_cols=False):
    """

    :param dfs: The data frames to merge, consisting of standard and "lista"
        (candidate preference) columns.
    :param lista_idxs_to_exclude: (Ballot) number (1...) of lists to exclude.
    :param return_overview_cols: Instructs to return the overview columns
        additionally. These will be described in the second member of the return
        tuple.

    :return: A single data frame if return_overview_cols is False. Otherwise
        a tuple of a data frame and a MergedListaInfo to help with navigating
        around.
    """
    print("merging \"lista\" columns...")
    lista_col_names = set()

    for df in dfs:
        for col in df.columns:
            if col.startswith("Lista nr"):
                lista_col_names.add(col)

    lista_col_names = sorted(lista_col_names, key=get_lista_col_idx)
    lista_col_names = [col for col in lista_col_names
                       if get_lista_col_idx(col)
                       not in lista_idxs_to_exclude]
    cols_to_keep = [_AREA_CODE_COLNAME] + lista_col_names

    if return_overview_cols:
        cols_to_keep += [
            _VALID_VOTES_COLNAME,
            _VOTERS_ELIGIBLE_TO_VOTE_COLNAME,
        ]

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

    if not return_overview_cols:
        return merged
    else:
        info = MergedDataInfo(
            _AREA_CODE_COLNAME,
            _VALID_VOTES_COLNAME,
            # fingers crossed it's the right one :)
            _VOTERS_ELIGIBLE_TO_VOTE_COLNAME,
            lista_columns=lista_col_names,
        )
        return merged, info
