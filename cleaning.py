import os.path
import pandas as pd
import numpy as np


def get_merged_data() -> pd.DataFrame:
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
    return df


def get_cleaned_data():
    if not os.path.exists("cleaned.csv"):
        df = get_merged_data()
        """ there is a mostly NaN row (someone left a total count in)
            --> remove!
        """
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
        df.to_csv("cleaned.csv", index=False)
    else:
        df = pd.read_csv("cleaned.csv")
    return df
