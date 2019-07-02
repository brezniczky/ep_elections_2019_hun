import os.path
import pandas as pd
import numpy as np
from cleaning import get_cleaned_data


def get_preprocessed_data(add_last_digits=True) -> pd.DataFrame:
    df = get_cleaned_data()

    columns = df.columns.copy()

    if add_last_digits:
        df["ld_Nevjegyzekben"] = df["Nevjegyzekben"] % 10
        df["ld_Ervenyes"] = df["Ervenyes"] % 10
        df["ld_Ervenytelen"] = df["Ervenytelen"] % 10
        for column in columns[-10:]:
            df[column] = df[column].astype(int)
            df["ld_" + column] = df[column] % 10

    return df
