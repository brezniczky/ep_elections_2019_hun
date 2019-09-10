"""
This is mainly doable in 2014.

The Jobbik party in that year has received a lot of >=100 votes, so we
can expect their last digit distribution to be more or less uniform
among such wards that all have >=100 votes and constitute a single settlement.

The Fidesz party has typically received even more votes.

The key concept is if the two last digit distributions were to be affected by
the same impact (say same miscalculation/rounding/"audio error" etc.) then
their "joint PMF" would not be free from coincidences (expect growing peaks
when one is tallied up with the other). Therefore the probability of their
entropy is likely to reduce.

And this we can hopefully catch with a reasonable accuracy.
"""
from HU.AndrasKalman.load import load_2014, plot_location
import pandas as pd
import numpy as np
from drdigit.digit_entropy_distribution import get_entropy, prob_of_entr
from collections import OrderedDict
from arguments import is_quiet, load_output, save_output


def save_results(df_fidesz_jobbik_joint, suspects):
    save_output(df_fidesz_jobbik_joint, "app7_fidesz_jobbik_joint.csv")
    save_output(suspects, "app7_suspects.csv")


def load_results():
    df_fidesz_jobbik_joint = load_output("app7_fidesz_jobbik_joint.csv")
    suspects = load_output("app7_suspects.csv")
    return df_fidesz_jobbik_joint, suspects


def generate_data():
    df_2014 = load_2014()
    df_Fidesz = df_2014[["Telepules", "Fidesz"]].copy()
    df_Jobbik = df_2014[["Telepules", "Jobbik"]].copy()
    df_Fidesz.columns = ["Telepules", "Fidesz_Jobbik"]
    df_Fidesz["ld"] = df_Fidesz["Fidesz_Jobbik"] % 10
    df_Jobbik.columns = ["Telepules", "Fidesz_Jobbik"]
    df_Jobbik["ld"] = df_Jobbik["Fidesz_Jobbik"] % 10
    df = pd.concat([df_Fidesz, df_Jobbik])

    agg_cols = OrderedDict([
        ("Fidesz_Jobbik", [min, len]),
        ("ld", get_entropy),
    ])

    df = df.groupby(["Telepules"]).aggregate(agg_cols)
    df.reset_index(inplace=True)
    df.columns = ["Telepules", "min_votes", "Korzet_count", "entropy"]

    def get_ent(row):
        return prob_of_entr(row.Korzet_count, row.entropy, seed=1235)

    df["p"] = df.apply(get_ent, axis=1)

    suspects = df[(df.p < 0.1) &
                  (df.min_votes >= 45) &
                  (df.Korzet_count >= 8 * 2)]
    return df, suspects


"""
Seed = 1234


In [29]: df[(df.p < 0.1) & (df.min_votes >= 50) & (df.Korzet_count >= 8 * 2)]
Out[29]:
          Telepules  min_votes  Korzet_count   entropy       p
333         Budaörs         52            40  2.095817  0.0816
364   Bátonyterenye         57            30  1.985231  0.0415
660         Edelény        120            16  1.771016  0.0824
1240        Kerepes        100            16  1.721402  0.0533
2101          Pomáz         54            32  1.956679  0.0144
2161         Pásztó         54            18  1.798106  0.0620
2195   Püspökladány        118            32  1.879748  0.0023
2440         Szeged         50           218  2.264374  0.0582
2500    Szigethalom         82            32  2.018515  0.0559


Seed = 1235

          Telepules  min_votes  Korzet_count   entropy       p
333         Budaörs         52            40  2.095817  0.0834
364   Bátonyterenye         57            30  1.985231  0.0448
660         Edelény        120            16  1.771016  0.0868
1240        Kerepes        100            16  1.721402  0.0541
2101          Pomáz         54            32  1.956679  0.0141
2161         Pásztó         54            18  1.798106  0.0639
2195   Püspökladány        118            32  1.879748  0.0020
2440         Szeged         50           218  2.264374  0.0586
2500    Szigethalom         82            32  2.018515  0.0560
"""


if __name__ == "__main__":
    if not "df" in globals():
        df, suspects = generate_data()
    if not is_quiet():
        for suspect in suspects.Telepules:
            plot_location(suspect)

    total_likely_denom = (int(round(1 / np.prod(suspects.p))))
    print("Total suspect likeliness (all of \"just these\" happening by chance):\n"
          "1 : %d" % total_likely_denom)
    print("Number of wards:", len(df))
    print("'Residual' unlikeliness:\n1 : %d" %
          int(round(total_likely_denom / len(df))))

    save_results(df, suspects)
    # TODO: test the above with truly uniformly generated values
