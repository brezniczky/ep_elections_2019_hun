"""
A short module to load the data I found on Kaggle.

Source URL:
https://www.kaggle.com/akalman/hungarian-parliamentary-elections-results

Thanks to Andras Kalman for carrying out the web scraping.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


filename_2014 = os.path.join(os.path.dirname(__file__),
                             "wards_2014-2018-04-19T16-32-08.json")


def ward_general_results_to_dict(ward_json):
    party_results = ward_json["general_list_results"]["party_results"]
    d = {
        party_result["party_name"]: party_result["num_of_votes"]
        for party_result in party_results
    }
    d.update({
        "district": ward_json["district"],
        "location": ward_json["location"],
        "number": ward_json["num"],
    })
    return d


def ward_general_results_to_dict(ward_json):
    party_results = ward_json["general_list_results"]["party_results"]
    d = {
        party_result["party_name"]: party_result["num_of_votes"]
        for party_result in party_results
    }
    d.update({
        "Korzet": ward_json["district"],
        "Telepules": ward_json["location"],
        "Szavazokor": int(ward_json["num"]),
    })
    return d


def ward_results_to_df(jo):
    dicts = [ward_general_results_to_dict(row) for row in jo]
    df = pd.DataFrame.from_dict(dicts)
    for col in df.columns:
        if col not in ["Korzet", "Telepules", "Szavazokor"]:
            df[col] = df[col].astype(int)
    return  df


def convert_colnames(df):
    """ Warning: inplace modification """
    party_name_map = {
        "FIDESZ-KDNP": "Fidesz",
        "JOBBIK": "Jobbik",
        # "LMP": "LMP",
        "EGYÜTT 2014": "Együtt",
    }

    df.columns = [
        party_name_map[column]
        if column in party_name_map
        else column
        for column in df.columns
    ]


def convert_settlement_names(df):

    def convert_settlement_name(name):
        if (name.startswith("Budapest") and
            name.endswith("ker.")
           ):
            return name.replace("ker.", " kerület")
        else:
            return name

    df.Telepules = \
        df.Telepules.apply(convert_settlement_name)


def load_2014():
    with open(filename_2014, "r") as f:
        jo = json.load(f)
    df = ward_results_to_df(jo)
    convert_colnames(df)
    convert_settlement_names(df)
    return df


if not "df_2014" in globals():
    df_2014 = load_2014()


def plot_location(location, df=df_2014):
    if df is None:
        df = df_2014
    bins = range(10)
    loc_digits = \
        df.loc[(df.Telepules==location)]["Fidesz"].values % 10
    plt.hist(loc_digits, bins=bins, alpha=0.5)
    loc_digits = \
        df.loc[(df.Telepules==location)]["Jobbik"].values % 10
    plt.hist(loc_digits, bins=bins, alpha=0.5)
    plt.title("%s, 2014, Fidesz vs. Jobbik" % location)
    plt.show()


if __name__ == "__main__":
    """
    for city in [
        # some suspects from a later 2014 check
        "Püspökladány",
        "Kerepes",
        "Kisújszállás",
        "Érd",
        "Budapest II. kerület",
        "Budapest XI. kerület",
        "Veresegyház",
        "Keszthely",
        "Mezőkövesd",
        "Balatonalmádi",
        "Szeged",
        "Nagykanizsa",
        "Kecskemét",
        "Diósd",
        "Szigethalom",
        "Ajka",
        "Kaposvár",
        # some suspects from later elections
        "Eger",
        "Tata",
        "Vecsés",
        "Sopron",
        "Balatonfüred",
        "Pécel",
        "Berettyóújfalu",
    ]:
        plot_location(city)
    """
