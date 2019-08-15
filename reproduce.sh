#!/bin/bash
#
# Script for recreating the results on *nix systems
#
# To the author's knowledge, any difference experienced should
# be down to numeric arithmetic subtleties.

virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
cd AndrasKalman
unzip hungarian-parliamentary-elections-results.zip
cd ..

python process_data.py

jupyter nbconvert --to html --execute report.ipynb
