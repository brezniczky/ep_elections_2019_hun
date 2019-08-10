"""
This scraper is created as even 2014 does not seem to be a reliable baseline.
It could help if 2010 were.


"""
from urllib.request import urlopen
from urllib.parse import urljoin
from time import sleep
import re
from bs4 import BeautifulSoup
from collections import OrderedDict
import pandas as pd
import os


# starts here
# https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/eredind.htm
# then we get to
# https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/tk.htm
# for capital districts and major municipalities (city with county rights etc.)
ROOT_URL_MAJOR = "https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/tk.htm"
ROOT_URL_SMALL = "https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/tk%s.htm"
# and letters ...
# https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/tka.htm

ROOT_URL_SMALL_CHARS = (
    "abcdefgh" +
    "ijklmnop" +
    "rstuvz"
)


"""
  e.g.
  <a href="01/003/szkkiv.htm" style="color: rgb(113,114,40)">Budapest III.ker.</a>
  <a href="01/015/szkkiv.htm" style="color: rgb(113,114,40)">Budapest XV.ker.</a>
  <a href="10/025/szkkiv.htm" style="color: rgb(113,114,40)">Eger</a>
"""
# MUNICIPALITY_PATTERN_MAJOR = "<a href=(\\d\\d/\\d\\d\\d/szkkiv.htm)>"
MUNICIPALITY_PATTERN = "<a href=.(\d\d/\d\d\d/szkkiv.htm). "

"""
  e.g.
  <a href="11/001/szkkiv.htm" style="color: rgb(113,114,40)">Abádszalók</a>
  
  effectively identical
"""
# MUNICIPALITY_PATTERN_SMALL = "<a href=(\\d\\d/\\d\\d\\d/szkkiv.htm)>"


"""
Turns to
https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/11/001/szkkiv.htm
"""
MUNICIPALITY_LINK_FMT = \
    "https://static.valasztas.hu/dyn/pv10/outroot/vdin1/hu/%s"

# e.g.
#
# Abádszalók
# <a href="jkv1.htm" style="color: rgb(113,114,40)">1</a>
# <a href="jkv5.htm" style="color: rgb(113,114,40)">5</a>
#
# Eger
# <a href="jkv8.htm" style="color: rgb(113,114,40)">8</a>
# <a href="jkv10.htm" style="color: rgb(113,114,40)">10</a>
#
WARD_PATTERN = "<a href=.(jkv\\d+.htm). "


SCRAPED_DIR = "2010/scraped"


""" 
My boilerplate stuff for scraping. Sorry no ScraPy, seemed daunting at first, 
guessed no time to learn. Beware copy-paste shared at this point. 
"""

def get_content(url):
    print("getting content", url)
    sleep(1)
    f = urlopen(url)
    content = f.read()
    status = f.getcode()
    print("status:", status)
    assert status == 200
    return content


def url_to_filename(url):
    return "".join(
        [c if not c in ":/.,;?&=" else "_"
         for c in url]
    )


def load_or_get_content(url):
    filename = os.path.join(SCRAPED_DIR, url_to_filename(url))
    if not os.path.exists(SCRAPED_DIR):
        os.mkdir(SCRAPED_DIR)

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    else:
        content = get_content(url).decode("iso-8859-2")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return content


def process_roots():
    minor_root_urls = [ROOT_URL_SMALL % char for char in ROOT_URL_SMALL_CHARS]
    # those listed at ROOT_URL_MAJOR are also contained in the alphabetic list
    root_urls = minor_root_urls

    municipality_links = []
    for root_url in root_urls:
        root_content = load_or_get_content(root_url)
        for hit in re.finditer(MUNICIPALITY_PATTERN, root_content):
            municipality_links.append(MUNICIPALITY_LINK_FMT % hit.group(1))
    return municipality_links


def process_municipalities(municipality_urls):
    ward_links = []
    for url in municipality_urls:
        content = load_or_get_content(url)
        for hit in re.finditer(WARD_PATTERN, content):
            ward_rel_link = hit.group(1)
            ward_link = urljoin(url, ward_rel_link)
            ward_links.append(ward_link)

    return ward_links


def process_wards(ward_urls):

    def strip_thousand_sep(num_str):
        return num_str.replace("\xa0", "")

    def _int(num_str):
        try:
            return int(strip_thousand_sep(num_str))
        except:
            import ipdb; ipdb.set_trace()

    df_dicts = []
    for url in ward_urls:
        print("dealing with", url)
        content = load_or_get_content(url)

        soup = BeautifulSoup(content, "html.parser")
        telepules_kor_row_b = soup.html.body.font.find_all("p")[0].font.b
        telepules_szavazokor_str = telepules_kor_row_b.contents[0]
        # Békéscsaba 1. sz. szavazókör

        iter = re.finditer(r"([^0-9]+) (\d+)", telepules_szavazokor_str)
        hit = iter.__next__()
        telepules, szavazokor = hit.group(1), hit.group(2)

        nevj_header = (soup.html.body.font.find_all("div")[4].table \
                       .find_all("tr")[1])
        nevj_row = (soup.html.body.font.find_all("div")[4].table \
                    .find_all("tr")[2])

        nevj_headers = [x.text for x in nevj_header.find_all("td")]
        nevj_values = [_int(x.text) for x in nevj_row.find_all("td")]

        # the below just didn't work ??
        # nevj_dict = dict(zip(nevj_headers, nevj_values))
        # UnboundLocalError: local variable 'dict' referenced before assignment
        # so:

        nevj_dict = {
            key: value for key, value in zip(nevj_headers, nevj_values)
        }

        nevjegyzekben_start = nevj_dict["A"]
        nevjegyzekben_transf = nevj_dict["B"]
        if len(nevj_dict) == 3:
            nevjegyzekben_end = nevj_dict["C"]
            nevjegyzekben_abroad = 0
        elif len(nevj_dict) == 5:
            nevjegyzekben_abroad = nevj_dict["D"]
            nevjegyzekben_end = nevj_dict["E"]
        else:
            assert False

        megjelentek = _int(soup.html.body.font.find_all("div")[5].table
                           .find_all("tr")[2].td.text)
        ervenyes = _int(soup.html.body.font.find_all("div")[6].table \
                        .find_all("tr")[2].find_all("td")[3].text)

        dict = {
            "Telepules": telepules,
            "Szavazokor": szavazokor,
            "Atjelentkezessel": nevjegyzekben_transf,
            "Kulkepviseleti": nevjegyzekben_abroad,
            "Nevjegyzekben": nevjegyzekben_end,
            "Megjelentek": megjelentek,
            "Ervenyes": ervenyes,
        }

        part_lista_div = soup.html.body.font.find_all("div")[7]
        tds = part_lista_div.table.tr.find_all("td")
        i = 4
        while i < len(tds):
            party_name = tds[i].text
            party_votes = _int(tds[i + 1].text)
            dict[party_name] = party_votes
            i += 3

        df_dicts.append(dict)

    df = pd.DataFrame(df_dicts)
    df = df.fillna(0)

    df["LMP"] = (df[" LMP"] +
                 df["Lehet Más a Politika"] +
                 df["LEHET MÁS A POLITIKA"])
    df["MDF"] += df["Magyar Demokrata Fórum"] + df['MAGYAR DEMOKRATA FÓRUM']
    df["Jobbik"] += df["JOBBIK"]
    df["MSZP"] += df["Magyar Szocialista Párt"] + df["MAGYAR SZOCIALISTA PÁRT"]
    df["Fidesz-KDNP"] += df['FIDESZ-KDNP'] + df['FIDESZ - KDNP']
    df["Civil Mozgalom"] += df["CIVIL MOZGALOM"] + df["CIVILEK"]
    df["Munkáspárt"] += df['MUNKÁSPÁRT']


    df.drop(columns=[" LMP", "Lehet Más a Politika", "LEHET MÁS A POLITIKA",
                     "Magyar Demokrata Fórum",
                     "JOBBIK",
                     "Magyar Szocialista Párt", "MAGYAR SZOCIALISTA PÁRT",
                     'MAGYAR DEMOKRATA FÓRUM',
                     'FIDESZ-KDNP', 'FIDESZ - KDNP', 'MUNKÁSPÁRT',
                     "CIVIL MOZGALOM", "CIVILEK"], inplace=True)

    df = df[[
        'Telepules', 'Szavazokor', 'Atjelentkezessel', 'Kulkepviseleti',
        'Nevjegyzekben', 'Megjelentek', 'Ervenyes',
        'Civil Mozgalom', 'Fidesz-KDNP', 'Jobbik',
        'MDF', 'MIÉP', 'MSZDP', 'MSZP',
        'Munkáspárt', 'Összefogás Párt BAZ M-i Lista', 'LMP'
    ]]
    df.to_csv("2010/hun_2010_general_elections_list.csv", index=False)


if __name__ == "__main__":
    municipality_urls = process_roots()
    ward_links = process_municipalities(municipality_urls)
    process_wards(ward_links)
