#!/bin/bash
#
# Script for recreating the results on *nix systems
#
# To the author's knowledge, any difference experienced should
# be down to numeric arithmetic subtleties.
#
# The first argment is passed to the script, intended use:
# pass --quick to do a quick test of the components integrating
# together, as well as whether there is any change in the "quick"
# output, suggesting the larger scale calculations are similarly
# unchanged ("quick" tells to use fewer iterations where it is
# regarded).
# It also instructs to skip the notebook updates.


# clean slate
rm -r .drdigit_cache
rm -r venv

virtualenv venv -p python3
echo `pwd` > venv/lib/python3.5/site-packages/drdigit.pth
source venv/bin/activate
pip install -r requirements.txt
cd HU/AndrasKalman
unzip -o hungarian-parliamentary-elections-results.zip
cd ../..

echo "Running app5_ent_in_top.py"
python HU/app5_ent_in_top.py --quiet $1
source venv/bin/activate
echo "Processing HU data"
python HU/process_data.py --quiet $1
echo "Processing PL data"
python PL/process_data.py --quiet $1 | tee PL_processing.log

if [ "$#" == 0 ] || [ $1 != "--quick" ]
then
    echo "updating HTML notebooks ..."
    jupyter nbconvert --to html --execute report.ipynb --ExecutePreprocessor.timeout=600
    jupyter nbconvert --to html --execute 'Poland 2019 EP Elections.ipynb' --ExecutePreprocessor.timeout=600
    jupyter nbconvert --to html --execute 'Austria 2019 EP Elections.ipynb' --ExecutePreprocessor.timeout=600
fi
