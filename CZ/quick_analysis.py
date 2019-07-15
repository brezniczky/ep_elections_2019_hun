import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Some parties with a more significant number of votes received:

ANO 2011
Ceska piratska strana
Ceska str.socialne demokrat.
Koalice STAN, TOP 09
Koalice Soukromnici, NEZ
Krest.demokr.unie-Cs.str.lid.
Obcanska demokraticka strana
Romska demokraticka strana
Svob.a pr.dem.-T.Okamura (SPD)


Party numbers:

1   Klub angazovanych nestraniku
2   Strana nezavislosti CR
3   CESTA ODPOVEDNE SPOLECNOSTI
4   Narodni socialiste
5   Obcanska demokraticka strana
6   ANO, vytrollime europarlament
7   Ceska str.socialne demokrat.
8   Romska demokraticka strana
9   Komunisticka str.Cech a Moravy
10  Koalice DSSS a NF
11  SPR-Republ.str.Csl. M.Sladka
12  Koalice Rozumni, ND
13  Volte Pr.Blok www.cibulka.net
14  NE-VOLIM.CZ
15  Pro Cesko
16  Vedci pro Ceskou republiku
17  Koalice CSNS, Patrioti CR
18  JSI PRO?Jist.Solid.In.pro bud.
19  PRO Zdravi a Sport
21  Moravske zemske hnuti
22  Ceska Suverenita
23  TVUJ KANDIDAT
24  HLAS
25  Koalice Svobodni, RC
26  Koalice STAN, TOP 09
27  Ceska piratska strana
28  Svob.a pr.dem.-T.Okamura (SPD)
29  ALIANCE NARODNICH SIL
30  ANO 2011
31  Agrarni demokraticka strana
32  Moravane
33  PRVNI REPUBLIKA
34  Demokraticka strana zelenych
35  BEZPECNOST,ODPOVEDNOST,SOLID.
36  Koalice Soukromnici, NEZ
37  Evropa spolecne
38  KONZERVATIVNI ALTERNATIVA
39  Krest.demokr.unie-Cs.str.lid.
40  Alternativa pro Cesk. rep.2017
"""


PARTY_RESULTS_FILENAME = "party_results.csv"
ANO_2011_NR = 30


def load_results():
    csv = pd.read_csv(PARTY_RESULTS_FILENAME)
    return csv[["region",
                "district",
                "municipality",
                "ward",
                "party_name",
                "party_nr",
                "total_votes",
                "perc_votes"]]


def plot_last_digits(df, party=""):
    if party:
        title = "Last digit distribution of %s" % party
    else:
        title = "Last digit distriubtion"
    last_digit = df["total_votes"] % 10
    plt.hist(last_digit)
    plt.show()



def show_ANO_plots(csv):
    # plot_last_digits(csv)
    csv = csv.copy()
    np.random.seed(1234)
    csv.distbnc = np.random.choice(10, len(csv))
    csv["total_votes_distbnc"] = csv.total_votes + csv.distbnc
    ANO_2011 = csv[csv["party_nr"] == ANO_2011_NR]
    # plot_last_digits(ANO_2011)
    ANO_2011_ABOVE_50 = csv[csv["total_votes_distbnc"] >= 50]
    plot_last_digits(ANO_2011_ABOVE_50)
    ANO_2011_ABOVE_70 = csv[csv["total_votes_distbnc"] >= 70]
    plot_last_digits(ANO_2011_ABOVE_70)
    ANO_2011_ABOVE_90 = csv[csv["total_votes_distbnc"] >= 90]
    plot_last_digits(ANO_2011_ABOVE_90)
    ANO_2011_ABOVE_100 = csv[csv["total_votes_distbnc"] >= 100]
    plot_last_digits(ANO_2011_ABOVE_100)
    ANO_2011_BELOW_100 = csv[csv["total_votes_distbnc"] < 100]
    plot_last_digits(ANO_2011_BELOW_100)


if __name__ == "__main__":
    if "long_csv" not in globals():
        long_csv = load_results()

    show_ANO_plots(long_csv)
