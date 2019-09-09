#!/bin/bash
#
# Script for recreating the results on *nix systems
#
# To the author's knowledge, any difference experienced should
# be down to numeric arithmetic subtleties.

# clean slate
rm -r .drdigit_cache
rm -r venv

virtualenv venv -p python3
echo `pwd` > venv/lib/python3.5/site-packages/drdigit.pth
source venv/bin/activate
pip install -r requirements.txt
cd AndrasKalman
unzip -o hungarian-parliamentary-elections-results.zip
cd ..

echo "Running app5_ent_in_top.py"
python HU/app5_ent_in_top.py --quiet
python HU/process_data.py --quiet
python PL/process_data.py --quiet | tee PL_processing.log

jupyter nbconvert --to html --execute report.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to html --execute 'Poland 2019 EP Elections.ipynb' --ExecutePreprocessor.timeout=600
jupyter nbconvert --to html --execute 'Austria 2019 EP Elections.ipynb' --ExecutePreprocessor.timeout=600
