import os.path
import pandas as pd
import numpy as np


def get_cleaned_data() -> pd.DataFrame:
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

    df.columns = [
        "Unnamed", "Megye", "Telepules", "Szavazokor", "Nevjegyzekben", "Megjelent",
        "Belyegzetlen", "Lebelyegzett", "Elteres megjelentektol", "Ervenytelen", "Ervenyes",
        "MSZP", "MKKP", "Jobbik", "Fidesz", "Momentum", "DK", "Mi Hazank", "Munkaspart", "LMP"
    ]

    # There is a mostly NAN line at the end of the Budapest sheet, remove it
    nan_line_idxs = np.where(np.isnan(df.Ervenyes))
    if len(nan_line_idxs) != 1 or (nan_line_idxs[0] != 1405):
        raise Exception("Only a certain NaN line was expected, please check the data.")
    df.drop(nan_line_idxs[0], inplace=True)

    # a bit of regret: this is not strictly part of the cleaned data,
    # maybe this function should be called get_preprocessed_data()
    # of course the project is already "beyond budget" (==0 :) )
    columns = df.columns.copy()

    for column in columns[-10:]:
        df[column] = df[column].astype(int)
        df["ld_" + column] = df[column] % 10
    df["ld_Nevjegyzekben"] = df["Nevjegyzekben"] % 10
    df["ld_Ervenytelen"] = df["Ervenytelen"] % 10

    return df
