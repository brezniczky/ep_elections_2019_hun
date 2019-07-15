"""
Only a quick check
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_cleaned_data():
    df = pd.read_csv("AT/endgueltiges_Ergebnis_inkl_WK_mit_Gemeindeergebnissen_EW19.csv")
    df = df[~df.GKZ.astype(str).str.endswith("00") &
            ~df.GKZ.astype(str).str.endswith("99")]
    return df


if __name__ == "__main__":
    df = get_cleaned_data()
    print(df)
    print(df.columns)
    col1 = df["ÖVP"]
    col2 = df["SPÖ"]
    col3 = df["FPÖ"]

    np.random.seed(1234)

    for limit in [100]: # [100, 500, 1000]:
        def s():
            return np.random.choice(range(10), len(df)) - 5

        ok = (col1 >= limit + s()) & (col2 >= limit + s()) & (col3 >= limit + s())

        plt.hist(col1[ok] % 10, alpha=0.5, bins=10)
        plt.title("ÖVP, min %d" % limit)
        plt.show()
        plt.hist(col2[ok] % 10, alpha=0.5, bins=10)
        plt.title("SPÖ, min %d" % limit)
        plt.show()
        plt.hist(col3[ok] % 10, alpha=0.5, bins=10)
        plt.title("FPÖ, min %d" % limit)
        plt.show()

    ok = (col1 >= 100) & (col2 >= 100) & (col3 >= 100)
