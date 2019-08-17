import pandas as pd
from functools import lru_cache
import os.path
from collections import OrderedDict


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


