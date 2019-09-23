This directory contains the first round results of the then two round parliamentary elections in Hungary, 2006.

The source files were downloaded from [www.valasztas.hu](www.valasztas.hu), the official Hungarian election website.

`scrape_2006.py` can be used to repeat this data collection step, use the virtual environment set up from the `requirements.txt` file in the root e.g. by the `reproduce.sh` script.

I.e. (from the repo root):

    $ source venv/bin/activate
    $ python HU/2006/scrape_2006.py

Intermediate files (on disk cache) should get created in the `HU/2006/scraped` directory, which should be deleted later in order to completely repeat the scraping again.

The resulting amalgamated data is in the `hun_2006_general_elections_list.csv` file.
