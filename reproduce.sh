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
python app5_ent_in_top.py
python explore_2018.py
python app6_comparative.py
python app7_fidesz_vs_jobbik.py
python app8_prob_of_twin_digits.py
python app9_overall_log_likelihood.py
python app14_digit_correlations_Hun.py

jupyter nbconvert --to html --execute report.ipynb
