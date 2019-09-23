"""
Scraper to retrieve the 2006 first round results of the Hungarian election.

This scraper is created as even 2010 does not seem to be a reliable baseline.
It could help if 2006 were :)

But, at least the magnitude of difference between different groups is reducing.
"""
from urllib.request import urlopen
from urllib.parse import urljoin
from time import sleep
import re
from bs4 import BeautifulSoup
import pandas as pd
import os


FINAL_COLUMNS = [
    "Telepules",
    "Szavazokor",
    "Nevjegyzekben",
    "Megjelentek",
    "Atjelentkezessel",
    "Kulkepviseleti",
    "Ervenyes",
    "CENTRUM",
    "FIDESZ-KDNP",
    "FKNP",
    "KERESZTÉNYDEMOKRATAPÁRT",
    "MAGYAR DEMOKRATA FÓRUM",
    "MCF ROMA ÖSSZEFOGÁS PÁRT",
    "MESZ",
    "MIÉP - JOBBIK",
    "MSZP",
    "MUNKÁSPÁRT",
    "MVPP",
    "SZDSZ",
    "ZÖLDEK PÁRTJA",
]


# at
# www.valasztas.hu
# choose "EN"
# then in the menu choose Elections and referendums/2006
# taking to
# https://static.valasztas.hu/parval2006/main_en.html
# choose "Results", getting to
# https://static.valasztas.hu/parval2006/en/08/8_0.html
# there choose
# "Result on the individual constituencies (IC)"
# taking to
# https://static.valasztas.hu/parval2006/en/08/8_0.html
# there choose "Back to 1st Round", but only right click it
# and choose "Open in new tab", and here we go:
# (otherwise it'd be an embedded frame, now we got an URL with ease)
# https://static.valasztas.hu/parval2006/outroot/vdin1/en/oevker.htm
#
# then unfortunately we might notice these are not too detailed
# whilst the Hungarian versions are, so replace "en" with "hu":
#
# https://static.valasztas.hu/parval2006/outroot/vdin1/hu/ejk0101.htm
#
# where ward-level links are actually listed at the bottom

# Root url:
ROOT_URL = "https://static.valasztas.hu/parval2006/outroot/vdin1/en/oevker.htm"
# find elements like
# <a href="ejk1001.htm" style="color: rgb(67,100,81)">01</a>
# <a href="ejk0101.htm" style="color: rgb(67,100,81)">01</a>
# then open these with the EN->HU switched root (explained above),
HU_ROOT_URL = "https://static.valasztas.hu/parval2006/outroot/vdin1/hu/oevker.htm"
# on these pages find elements like
# <a href="01/001/jkv1.htm" style="color: rgb(67,100,81)">1</a>
# <a href="01/001/jkv10.htm" style="color: rgb(67,100,81)">10</a>
# <a href="10/025/jkv1.htm" style="color: rgb(67,100,81)">1</a>
# <a href="10/025/jkv10.htm" style="color: rgb(67,100,81)">10</a>
# then open these and whatever, there's the data!

# <a href="ejk1001.htm" style="color: rgb(67,100,81)">01</a>
CONSTITUENCY_PATTERN = "<a href=.([a-z]{3}\d{4}.htm)."

WARD_PATTERN = "<a href=.(\d\d/\d\d\d/[a-z]{3}\d{1,2}.htm)."

SCRAPED_DIR = "HU/2006/scraped"

"""
My boilerplate stuff for scraping. Sorry no ScraPy, seemed daunting at first,
guessed no time to learn. Beware copy-paste shared at this point.
"""


def get_content(url):
    print("getting content", url)
    sleep(0.25)
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
    root_content = load_or_get_content(ROOT_URL)
    constituency_links = []
    for hit in re.finditer(CONSTITUENCY_PATTERN, root_content):
        constituency_links.append(urljoin(HU_ROOT_URL, hit.group(1)))
    return constituency_links


def process_constituencies(constituency_urls):
    ward_links = []
    for url in constituency_urls:
        content = load_or_get_content(url)
        for hit in re.finditer(WARD_PATTERN, content):
            ward_rel_link = hit.group(1)
            ward_link = urljoin(url, ward_rel_link)
            ward_links.append(ward_link)

    return ward_links


def strip_thousand_sep(num_str):
    return num_str.replace("\xa0", "")


def _int(num_str):
    try:
        return int(strip_thousand_sep(num_str))
    except:
        import ipdb
        ipdb.set_trace()


def get_ward_data(content):
    # almost the same as for the HU/2010 version

    # TODO: might be worth a refactor

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
    nevjegyzekben_abroad = 0
    nevjegyzekben_refused = 0

    print(nevjegyzekben_start, nevjegyzekben_transf)

    if set(nevj_dict.keys()) == set(["A", "B", "C"]):
        nevjegyzekben_end = nevj_dict["C"]
        nevjegyzekben_abroad = 0
    elif set(nevj_dict.keys()) == set(["A", "B", "C", "D", "E"]):
        nevjegyzekben_abroad = nevj_dict["D"]
        nevjegyzekben_end = nevj_dict["E"]
    elif set(nevj_dict.keys()) == set(["A", "B", "C", "F"]):
        nevjegyzekben_end = nevj_dict["C"]
        nevjegyzekben_refused = nevj_dict["F"]
    elif set(nevj_dict.keys()) == set(["A", "B", "C", "D", "E", "F"]):
        nevjegyzekben_abroad = nevj_dict["D"]
        nevjegyzekben_end = nevj_dict["E"]
        nevjegyzekben_refused = nevj_dict["F"]
    else:
        print(nevj_dict)
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

    return dict


def process_wards(ward_urls):

    if not os.path.exists("2006_HU_temp.csv"):
        df_dicts = []
        for url in ward_urls:
            print("dealing with", url)
            content = load_or_get_content(url)

            try:
                dict = get_ward_data(content)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(e)
                dict = get_ward_data(content)

            df_dicts.append(dict)

        df = pd.DataFrame(df_dicts)
        df = df.fillna(0)

        df.to_csv("2006_HU_temp.csv", index=False)
    else:
        df = pd.read_csv("2006_HU_temp.csv")

    translations = {
        "FIDESZ-KDNP": ["FIDESZ - KDNP", "FIDESZ KDNP",
                        "FIDESZ-KDNP", "FIDESZ-MPSZ,KDNP",
                        "FIDESZ-MPSZ-KDNP"],
        "FKNP": ["FKNP", "FÜGGETLEN KISGAZDA NEMZETI E P",
                 "FÜGGETLEN KISGAZDAPÁRT"],
        "KERESZTÉNYDEMOKRATAPÁRT": ["KERESZTÉNYDEMOKRATAPÁRT",
                                    "Kereszténydemokratapárt"],
        "MAGYAR DEMOKRATA FÓRUM": ["MDF MAGYAR DEMOKRATA FÓRUM",
                                   "MAGYAR DEMOKRATA FÓRUM",
                                   "MDF"],
        "MCF ROMA ÖSSZEFOGÁS PÁRT": [
            "MCF ROMA ÖSSZEFOGÁS PÁRT",
            "MCF Roma Összefogás Párt",
            "ROMA ÖSSZEFOGÁS",
            "ROMA ÖSSZEFOGÁS PÁRT",
            "ROMA ÖSZEFOGÁS PÁRT",
        ],
        "MSZP": ["MAGYAR SZOCIALISTA PÁRT",
                 "MSZP",
                 "Magyar Szocialista Párt"],
        "CENTRUM": ["CENTRUM"],
        "SZDSZ": [
            "SZDSZ",
            "SZDSZ -A MAGYAR LIBERÁLIS PÁRT",
            "SZDSZ A MAGYAR LIBERÁLIS PÁRT",
            "SZDSZ a Magyar Liberális Párt",
            "SZDSZ-A MAGYAR LIBERÁLIS PÁRT",
            "Szabad Demokraták Szövetsége",
        ],
        "MIÉP - JOBBIK": [
            "MIÉP - JOBBIK",
            "MIÉP- Jobbik",
            "MIÉP-JOBBIK",
            "MIÉP-Jobbik",
            "MIÉP-Jobbik a Harmadik Út",
        ],
        "MUNKÁSPÁRT": [
            "MUNKÁSPÁRT",
            "MUNKÁSPÁRT 2006",
            "Munkáspárt",
        ]
    }

    import ipdb; ipdb.set_trace()

    for key, source_cols in translations.items():
        cols_to_sum = [df[col_name] for col_name in source_cols]
        total = cols_to_sum[0]
        for i in range(1, len(cols_to_sum)):
            total += cols_to_sum[i]
        df[key] = total
        to_drop = set(source_cols)
        to_drop = to_drop - set([key])
        df.drop(columns=to_drop, inplace=True)

    assert len(df.columns) == len(FINAL_COLUMNS)

    df = df[FINAL_COLUMNS]

    df.to_csv("HU/2006/hun_2006_general_elections_list.csv", index=False)


if __name__ == "__main__":
    constituency_urls = process_roots()
    ward_links = process_constituencies(constituency_urls)
    process_wards(ward_links)

    # #
    # urls = [
    #     "https://static.valasztas.hu/parval2006/outroot/vdin1/hu/02/015/jkv1.htm",
    #     "https://static.valasztas.hu/parval2006/outroot/vdin1/hu/05/202/jkv89.htm",
    # ]
    # for url in urls:
    #     content = load_or_get_content(url)
    #     dict = get_ward_data(content)
    #     import pprint
    #     for k in sorted(dict):
    #         print(k, dict[k])
