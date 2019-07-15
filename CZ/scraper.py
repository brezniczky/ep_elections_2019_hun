"""
A very simple scraper with a few delays and retries.
"""

"""
https://www.volby.cz/pls/ep2019/ep13?xjazyk=EN
-> municipalities:
https://www.volby.cz/pls/ep2019/ep133?xjazyk=EN&xnumnuts=2101
-> wards:
https://www.volby.cz/pls/ep2019/ep134?xjazyk=EN&xnumnuts=2101&xobec=529303
-> ward 1
https://www.volby.cz/pls/ep2019/ep1311?xjazyk=EN&xobec=529303&xokrsek=1&xvyber=2101

"""
from urllib.request import urlopen
from time import sleep
import re
import pandas as pd
import os


ROOT_URL = "https://www.volby.cz/pls/ep2019/ep13?xjazyk=EN"

DISTRICT_PATTERN = "<a href=\"ep133\\?xjazyk=EN&amp;xnumnuts=(\\d+)\">"

# turns to https://www.volby.cz/pls/ep2019/ep133?xjazyk=EN&xnumnuts=7201
DISTRICT_LINK = "https://www.volby.cz/pls/ep2019/ep133?xjazyk=EN&xnumnuts=%s"

# want: ep134?xjazyk=EN&xnumnuts=2101&xobec=529303
# "<a href=\"ep134\?xjazyk=EN&amp;xnumnuts=(\d+)&amp;xobec=(\d+)"
MUNICIPALITY_PATTERN = "<a href=\"ep134\\?xjazyk=EN&amp;xnumnuts=(\\d+)&amp;xobec=(\\d+)\">"

# turns to https://www.volby.cz/pls/ep2019/ep134?xjazyk=EN&xnumnuts=2101&xobec=529303
MUNICIPALITY_LINK = "https://www.volby.cz/pls/ep2019/ep134?xjazyk=EN&xnumnuts=%s&xobec=%s"
MUNICIPALITY_LINKS_FILENAME = "municipality_links.csv"


# WARD_LINK = "https://www.volby.cz/pls/ep2019/ep134?xjazyk=EN&xnumnuts=%s&xobec=%s"
# e.g. <a href="ep1311?xjazyk=EN&amp;xobec=564028&amp;xokrsek=1&amp;xvyber=5103">1</a>
WARD_PATTERN = "<a href=\"ep1311\\?xjazyk=EN&amp;xobec=(\\d+)&amp;xokrsek=(\\d+)&amp;xvyber=(\\d+)\">"
# turns to:
# https://www.volby.cz/pls/ep2019/ep1311?xjazyk=EN&xobec=564028&xokrsek=1&xvyber=5103
WARD_LINKS_FILENAME = "ward_links.csv"
WARD_LINK = "https://www.volby.cz/pls/ep2019/ep1311?xjazyk=EN&xobec=%s&xokrsek=%s&xvyber=%s"


PARTY_RESULTS_FILENAME = "party_results.csv"


WARD_INFO_REGION_PATTERN = "Region: ([^<]+)\n"
WARD_INFO_DISTRICT_PATTERN = "District: ([^<]+)\n"
WARD_INFO_MUNICIPALITY_PATTERN = "Municipality: ([^<]+)\n"
WARD_INFO_WARD_PATTERN = "Ward: ([^<]+)\n"

PARTY_RESULT_PATTERN = (
    "<tr>\n" +
    "<td class=\"cislo\" headers=\"[^\"]+\">(\\d+)</td>\n" +                 # party number
    "<td headers=\"[^\"]+\">([^<]+)</td>\n" +                                # party name
    "<td class=\"cislo\" headers=\"[^\"]+\">([^<]+)</td>\n" +                # total
    "<td class=\"cislo\" headers=\"[^\"]+\">([^<]+)</td>\n"                  # %
    ""
)


# Examples

"""
<tr>
<td class="cislo" headers="t1sa1 t1sb1">1</td>
<td headers="t1sa1 t1sb2">Klub angazovanych nestraniku</td>
<td class="cislo" headers="t1sa2 t1sb3">0</td>
<td class="cislo" headers="t1sa2 t1sb4">0.00</td>
<td class="hidden_td" headers="t1sa3">-</td>
</tr>
"""

"""
<tr>
<td class="cislo" headers="t2sa1 t2sb1">22</td>
<td headers="t2sa1 t2sb2">Ceska Suverenita</td>
<td class="cislo" headers="t2sa2 t2sb3">0</td>
<td class="cislo" headers="t2sa2 t2sb4">0.00</td>
<td class="hidden_td" headers="t2sa3">-</td>
</tr>
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
    filename = os.path.join("scraped", url_to_filename(url))
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    else:
        content = get_content(url).decode("iso-8859-2")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return content


def process_root():
    root_content = load_or_get_content(ROOT_URL)
    district_links = []
    for hit in re.finditer(DISTRICT_PATTERN, root_content):
        district_links.append(DISTRICT_LINK % hit.group(1))
    return district_links


def load_or_extract_municipality_links():

    def process_districts():
        municipality_links = []
        for link in district_links:
            content = load_or_get_content(link)
            for hit in re.finditer(MUNICIPALITY_PATTERN, content):
                n1, n2 = (hit.group(1), hit.group(2))
                municipality_links.append(MUNICIPALITY_LINK % (n1, n2))
        return municipality_links

    if os.path.exists(MUNICIPALITY_LINKS_FILENAME):
        df = pd.read_csv(MUNICIPALITY_LINKS_FILENAME)
        municipality_links = list(df["link"])
    else:
        municipality_links = process_districts()
        df = pd.DataFrame(dict(link=municipality_links))
        df.to_csv(MUNICIPALITY_LINKS_FILENAME, index=False)
    return municipality_links


def load_or_extract_ward_links():
    # e.g. from
    # https://www.volby.cz/pls/ep2019/ep134?xjazyk=EN&xnumnuts=5103&xobec=564028
    def process_municipalities():
        ward_links = []
        for link in municipality_links:
            content = load_or_get_content(link)
            for hit in re.finditer(WARD_PATTERN, content):
                ward_links.append(
                    WARD_LINK % tuple(hit.group(k) for k in [1, 2, 3])
                )
            print("found %d ward links so far ..." % len(ward_links))
        return ward_links

    if os.path.exists(WARD_LINKS_FILENAME):
        df = pd.read_csv(WARD_LINKS_FILENAME)
        ward_links = list(df["link"])
    else:
        ward_links = process_municipalities()
        df = pd.DataFrame(dict(link=ward_links))
        df.to_csv(WARD_LINKS_FILENAME, index=False)
    return ward_links


def get_ward_content_data(content):
    party_result_list = []

    def get_field(pattern):
        nonlocal content
        match = re.search(pattern, content)
        return match.group(1) if match else "n/a"

    region = get_field(WARD_INFO_REGION_PATTERN)
    district = get_field(WARD_INFO_DISTRICT_PATTERN)
    municipality = get_field(WARD_INFO_MUNICIPALITY_PATTERN)
    ward = get_field(WARD_INFO_WARD_PATTERN)

    for hit in re.finditer(PARTY_RESULT_PATTERN, content, flags=re.MULTILINE):
        party_nr, party_name, total_votes, perc_votes = \
            tuple(hit.group(k) for k in [1, 2, 3, 4])
        party_result_list.append(dict(
            region=region,
            district=district,
            municipality=municipality,
            ward=ward,
            party_nr=party_nr,
            party_name=party_name,
            total_votes=total_votes,
            perc_votes=perc_votes,
        ))

    return party_result_list


def load_or_extract_party_results():

    def process_wards():
        party_results_list = []
        i = 0
        for link in ward_links:
            content = load_or_get_content(link)

            party_results_list.extend(
                get_ward_content_data(content)
            )

            i += 1
            if i % 30 == 0:
                print(party_results_list[-30:])
        return pd.DataFrame(party_results_list)

    if os.path.exists(PARTY_RESULTS_FILENAME):
        party_results = pd.read_csv(PARTY_RESULTS_FILENAME)
    else:
        party_results = process_wards()
        party_results.to_csv(PARTY_RESULTS_FILENAME, index=False)
    return party_results


if __name__ == "__main__":
    if "district_links" not in globals():
        district_links = process_root()
    if "municipality_links" not in globals():
        municipality_links = load_or_extract_municipality_links()
    if "ward_links" not in globals():
        ward_links = load_or_extract_ward_links()
    if "party_results" not in globals():
        party_results = load_or_extract_party_results()
