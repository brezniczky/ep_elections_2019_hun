## Welcome!

This repository hosts a best effort analysis providing some degree of suspicion 
about recent Hungarian elections.
It seems way more than was expected to be detectable by the author.

The script eventually should be migrated to a Kaggle kernel, but it's really 
fragmented and slow at this point.

All of the data directly involved can be found there:

2014 data, also increasingly used as 2018 data: 
https://www.kaggle.com/akalman/hungarian-parliamentary-elections-results

Alternative 2018 data (less verbose): 
https://www.kaggle.com/brezniczky/hungarian-parliamentary-elections-2018-dataset
2019 data: https://www.kaggle.com/brezniczky/hungarian-ep-elections-data-2019

The repository contains data from other countries which may later provide some 
degree of a reference/baseline.

The resulting analysis can be accessed e.g. here:
https://nbviewer.jupyter.org/github/brezniczky/ep_elections_2019_hun/blob/master/report.ipynb

It is yet to be finished properly (humps and bumps here and there)/peer 
reviewed, however, apparently it is urgent to publish it as the next sizeable 
major elections (Hungarian local elections) are around the bend, in October, 
then we might have to live with the consequences of ignorance for years.

The report should be reproducible (as of about 2 days before the time writing), 
it takes a while (a couple of hours primarily due to early and irrelevant 
simulations - I know... should untie the dependencies), but still, you know, 
reproducible :)

For this, a bash script is provided: reproduce.sh.
Run reproduce.sh e.g. with

```bash reproduce.sh```

in the root directory of the clone of this repository, and wait... and close the 
chart windows occasionally popping up. (Sorry #n+1.)
The report.html file should get updated in the root directory.

A few steps were made with part of the implementation towards more reusability,
perhaps a Python package (there are lots of interesting and likely a lot more
seriously worked out means of testing for fraud around - there would be more
than a few things to include).

There's also a quick demo of what there is as the foundations of just my tests,
looking at the 2019 Polish EP data:

[A quick assessment of the 2019 Polish EP election results](
https://htmlpreview.github.io/?https://github.com/brezniczky/ep_elections_2019_hun/blob/master/Poland%202019%20EP%20Elections.html
) 
