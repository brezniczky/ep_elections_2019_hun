import pandas as pd
import numpy as np
import json
from arguments import output_exists, load_output, save_output


def _translate_capital_district_name(col):
    col = col.copy()
    extr = col.str.extract(r"^(Budapest [^.]+)\.")
    is_hit = ~pd.isnull(extr[0])
    col.loc[is_hit] = extr[0][is_hit].str.cat(['. kerület'] * sum(is_hit))
    return col


def get_merged_data() -> pd.DataFrame:
    if not output_exists("merged.csv"):
        filename = r'HU/EP_2019_szavaz_k_ri_eredm_ny.xlsx'
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
        save_output(df, "merged.csv")
    else:
        df = load_output("merged.csv")
    return df


def get_cleaned_data():
    if not output_exists("cleaned.csv"):
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
        save_output(df, "cleaned.csv")
    else:
        df = load_output("cleaned.csv")

    df["Telepules"] = _translate_capital_district_name(df["Telepules"])
    return df


def get_2014_cleaned_data():
    """
    # TODO: report bogus nature

    Warning: bogus data, e.g.

    [29]:
    {'received_envelopes': '0',
     'locals_voted': '497',
     'total_registered': '1753',
     'total_voted': '0',
     'invalid_pages': '9',
     'valid_pages': '1331',
     'stamped_pages_in_urn_and_envelopes': '1340'}

    probably the total_voted = 0 is incorrect

    """
    with open("AndrasKalman/wards_2014-2018-04-19T16-32-08.json", "r") as f:
        jd = json.load(f)

    dicts = []

    name_translation = {
        'A HAZA NEM ELADÓ': "A Haza Nem Eladó",
        'EGYÜTT 2014': "Együtt 2014",
        'FIDESZ-KDNP': "Fidesz-KDNP",
        'FKGP': "FKGP",
        "JESZ": "JESZ",
        'JOBBIK': "Jobbik",
        'KTI': "KTI",
        'LMP': "LMP",
        'MCP': "MCP",
        'MSZP-EGYÜTT-DK-PM-MLP': "MSZP-Együtt-DK-PM-MLP",
        'MUNKÁSPÁRT': "Munkáspárt",
        'SEM': "SEM",
        'SMS': "SMS",
        'SZOCIÁLDEMOKRATÁK': "Szociáldemokraták",
        'ZÖLDEK': "Zöldek",
        'ÖP': "ÖP",
        'ÚDP': "ÚDP",
        'ÚMP': "ÚMP"
    }

    for row in jd:
        party_list_summary = \
            row["general_list_results"]["party_list_summary"]
        dict = {
            "Nevjegyzekben":
                # 28 has "locals_registered", investigate why
                int(party_list_summary["total_registered"])
                if "total_registered" in party_list_summary else
                int(party_list_summary["locals_registered"]),
            "Ervenyes":
                int(party_list_summary["valid_pages"]),
            "Telepules": row["location"],
            "Szavazokor": row["num"],
        }
        for party_result in row["general_list_results"]["party_results"]:
            dict[name_translation[party_result["party_name"]]] = \
                int(party_result["num_of_votes"])
        dicts.append(dict)

    df = pd.DataFrame(dicts)
    df["Telepules"] = _translate_capital_district_name(df["Telepules"])
    return df


def get_2018_cleaned_data():
    """
    :return:


    Warning: bogus data, e.g.

    [29]:
    {'received_envelopes': '0',
     'locals_voted': '497',
     'total_registered': '1753',
     'total_voted': '0',
     'invalid_pages': '9',
     'valid_pages': '1331',
     'stamped_pages_in_urn_and_envelopes': '1340'}

    probably the total_voted = 0 is incorrect

    """
    with open("AndrasKalman/wards_2018-2018-04-16T08-44-21.json", "r") as f:
        jd = json.load(f)

    dicts = []

    # TODO
    # name_translation = {
    #     'A HAZA NEM ELADÓ': "A Haza Nem Eladó",
    #     'EGYÜTT 2014': "Együtt 2014",
    #     'FIDESZ-KDNP': "Fidesz-KDNP",
    #     'FKGP': "FKGP",
    #     "JESZ": "JESZ",
    #     'JOBBIK': "Jobbik",
    #     'KTI': "KTI",
    #     'LMP': "LMP",
    #     'MCP': "MCP",
    #     'MSZP-EGYÜTT-DK-PM-MLP': "MSZP-Együtt-DK-PM-MLP",
    #     'MUNKÁSPÁRT': "Munkáspárt",
    #     'SEM': "SEM",
    #     'SMS': "SMS",
    #     'SZOCIÁLDEMOKRATÁK': "Szociáldemokraták",
    #     'ZÖLDEK': "Zöldek",
    #     'ÖP': "ÖP",
    #     'ÚDP': "ÚDP",
    #     'ÚMP': "ÚMP"
    # }

    for row in jd:
        party_list_summary = \
            row["general_list_results"]["party_list_summary"]
        dict = {
            "Nevjegyzekben":
            # 28 has "locals_registered", investigate why
                int(party_list_summary["total_registered"])
                if "total_registered" in party_list_summary else
                int(party_list_summary["locals_registered"]),
            "Ervenyes":
                int(party_list_summary["valid_pages"]),
            "Telepules": row["location"],
            "Szavazokor": row["num"],
        }
        for party_result in row["general_list_results"]["party_results"]:
            dict[party_result["party_name"]] = \
                int(party_result["num_of_votes"])
        dicts.append(dict)

    df = pd.DataFrame(dicts)
    df["Telepules"] = _translate_capital_district_name(df["Telepules"])
    return df


def get_2010_cleaned_data():
    df = pd.read_csv("2010/hun_2010_general_elections_list.csv")
    df["Telepules"] = _translate_capital_district_name(df["Telepules"])
    return df
