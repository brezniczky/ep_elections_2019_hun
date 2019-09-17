2019 Hungarian EP Elections Analysis
====================================

This repository hosts a best effort analysis providing a degree of suspicion
about recent Hungarian elections.

It seems way greater than was expected to be detectable by the author.

It is yet to be finished properly (humps and bumps here and there)/peer
reviewed, however, apparently it is urgent to make it public as the next
sizeable major elections (Hungarian local elections) are around the bend, in
mid-October, then we might have to live with the consequences of ignorance for
years.


Requirements
------------

The analysis was created in and essentially requires Python 3.5.

Hungarian elections
-------------------

#### Analysis

[Irregularities in recent (2018-2019) Hungarian election data](https://htmlpreview.github.io/?https://github.com/brezniczky/ep_elections_2019_hun/blob/master/Hungary%202019%20EP%20Elections.html)

#### Data involved (Kaggle links)

The source of 2014 data, also increasingly used as 2018 data:

https://www.kaggle.com/akalman/hungarian-parliamentary-elections-results

Alternative 2018 data (less verbose, insufficient for fingerprint plots - my
bad):

https://www.kaggle.com/brezniczky/hungarian-parliamentary-elections-2018-dataset

2019 data:

https://www.kaggle.com/brezniczky/hungarian-ep-elections-data-2019


Reproduction
------------

To obtain a copy, clone this github repository e.g.

    $ git clone https://github.com/brezniczky/ep_elections_2019_hun

and enter the resulting directory, assuming the above command, simply

    $ cd ep_elections_2019_hun

To reproduce its results and recompile the resulting document, under a linux
compatible shell, simply run

    $ bash reproduce.sh

from the directory just entered.

This is time consuming, about 2 hours can easily be taken up.
The `report.html` file should get updated in the repository directory.

The required packages should be automatically downloaded installed by the
reproduce.sh script which sets up a virtual environment in the venv
subdirectory. You can use it by

    $ source venv/bin/activate

to play with the individual steps, re-run approaches.
Approaches 1-4 (`app1_....py`...`app4_....py`) are completely deprecated and
excluded from the results.


Plans
-----

The script/notebook eventually should soon be migrated to a Kaggle kernel, but
it's really fragmented and slow at this point.


DrDigit Package
---------------

A digit doctoring detection package is being extracted from the core of the
analysis (work in progress).
Whether or not the supposed findings of the analysis themselves will stand, I
hope this may spark further work on perhaps different countries' data, can grow
into something useful or at least fun.

The package can be installed via pip:

    $ pip install drdigit-brezniczky

More information can be found on the package repo or via the console help from
a Python prompt

    $ import drdigit as drd
    $ help(drd)


Polish analysis and data
------------------------

There is a short assessment hosted on Kaggle, together with the respective data.
This demonstrates the use of the package on Polish data. The actuality of this
is that the next Polish election is set to take place on the exact same day as
the Hungarian one.

[Poland 2019 EP elections doctoring quick check](
https://www.kaggle.com/brezniczky/poland-2019-ep-elections-doctoring-quick-check/
)

At the time writing the initial assessment steps are yet to be extracted there,
you can find more info by looking into the [PL/process_data.py](
https://github.com/brezniczky/ep_elections_2019_hun/blob/master/PL/process_data.py
) script.

The data can be found in the preprocessed (better previewable) form on Kaggle,
or here the xlsx is in the [PL directory](
https://github.com/brezniczky/ep_elections_2019_hun/tree/master/PL
)


Austrian analysis
-----------------

There is a quick analysis on the Austrian data as well.
At the minute it's yet to be put on Kaggle.

[A quick assessment of the 2019 Austrian EP election results](
https://nbviewer.jupyter.org/github/brezniczky/ep_elections_2019_hun/blob/master/Austria%202019%20EP%20Elections.ipynb
)

It is also hosted on [Kaggle](https://www.kaggle.com/brezniczky/austria-2019-ep-elections-doctoring-quick-check), you can fork it, play with it, delve into the devilish details - you know :)


Data
----

The repository contains data from other countries which may later provide some
degree of a reference/baseline:

| Election | Repo data | Kaggle link |
| -------- | --------- | ----------- |
| Austria 2019 EP| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/AT)|[Link](https://www.kaggle.com/brezniczky/austrian-ep-elections-data-2019)|
| Czechia 2019 EP| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/CZ)||
| Germany 2019 EP (Berlin, Munich also for 2014)| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/DE)||
| Hungary 2010 General| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/HU/2010)||
| Hungary 2014-2018 General| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/HU/AndrasKalman)|[Link](https://www.kaggle.com/akalman/hungarian-parliamentary-elections-results)|
| Hungary 2018 General (alternative)| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/HU/2018) | [Link](https://www.kaggle.com/brezniczky/hungarian-parliamentary-elections-2018-dataset)|
| Hungary 2019 EP| [Link](https://github.com/brezniczky/ep_elections_2019_hun/blob/master/HU/EP_2019_szavaz_k_ri_eredm_ny.xlsx)|[Link](https://www.kaggle.com/brezniczky/hungarian-ep-elections-data-2019)|
| Poland 2019 EP| [Link](https://github.com/brezniczky/ep_elections_2019_hun/tree/master/PL)|[Link](https://www.kaggle.com/brezniczky/2019-european-parliament-election-in-poland-data)|


Contributions
-------------

Are always welcome, there is a DrDigit package Gitter room:
https://gitter.im/drdigit/community where we can discuss what you are
interested in, or just feel free to open/comment on an issue or open a pull
request!

Remark: I am trying to mostly comply with PEP8 here but it is not yet
ensured via CI.


Acknowledgements
----------------

Many thanks so far for the data and the inspiration to Miklós Vázsonyi, for the data to András Kálmán and Florian
Stagliano!
