# Overall robustness settings for now
# Bit of a parasite: replaces the cleaning.py exports with an additional
# processing phase once imported.

import warnings
from numpy.random import seed, choice
from arguments import get_default_output_dir, add_output_dir_postfix, is_quiet
import HU.cleaning as cl



"""
TODO: this switch should move to arguments.py but before then the overarching
      Bash script should be rewritten in Python ... it is just not a 21st
      century language e.g. see
https://unix.stackexchange.com/questions/290146/multiple-logical-operators-a-b-c-and-syntax-error-near-unexpected-t
"""
# Set this to True to try out the process with downsampled data
USE_DOWNSAMPLING = False

# How much of the original municipalities to randomly select
ROBUSTNESS_PERCENTAGE = 0.9
# Random seed used for the municipality selection
ROBUSTNESS_SEED = 1234


_ALREADY_APPLIED = False
_ALLOWED_MUNICIPALITIES = []
_EXCLUDED_MUNICIPALITIES = []


def _draw_allowed_municipalities():
    global _ALLOWED_MUNICIPALITIES
    global _EXCLUDED_MUNICIPALITIES

    df = cl.get_cleaned_data()
    all = sorted(list(set(df["Telepules"].values)))
    print("all:", all)
    seed(ROBUSTNESS_SEED)

    all_indexes = range(len(all))
    included_indexes = choice(range(len(all)),
                              int(round(ROBUSTNESS_PERCENTAGE * len(all))),
                              replace=False)
    excluded_indexes = set(all_indexes).difference(included_indexes)
    _ALLOWED_MUNICIPALITIES = list([all[i] for i in included_indexes])
    _EXCLUDED_MUNICIPALITIES = list([all[i] for i in excluded_indexes])

    _ALLOWED_MUNICIPALITIES = sorted(_ALLOWED_MUNICIPALITIES)
    _EXCLUDED_MUNICIPALITIES = sorted(_EXCLUDED_MUNICIPALITIES)


def _apply_downsampling_hooks():

    def downsample_Hungarian_data(df, f):
        df = df[df["Telepules"].isin(_ALLOWED_MUNICIPALITIES)]
        return df

    cl.add_processing_hook(downsample_Hungarian_data)


def init_downsampling():

    global _ALREADY_APPLIED
    if _ALREADY_APPLIED:
        warnings.warning("Filtering to verify robustness should be applied "
                         "only once (init_downsampling was called multiple "
                         "times). This call has no effect.")
        return

    _ALREADY_APPLIED = True
    _draw_allowed_municipalities()
    _apply_downsampling_hooks()

    file_spec_part = "_robustness_s_%d_p_%f" % (ROBUSTNESS_SEED,
                                                ROBUSTNESS_PERCENTAGE)
    valid_in_num = "0123456789eE-abcdefghijklmnopqrstuvwxyz"
    file_spec_part = "".join([k if k in valid_in_num else "_"
                              for k in file_spec_part])
    add_output_dir_postfix(file_spec_part)

    if is_quiet():
        print("Downsampled to %d municipalities" % len(_ALLOWED_MUNICIPALITIES))
    else:
        print("Allowed municipalities (%d): %s" % (
            len(_ALLOWED_MUNICIPALITIES),
            ",".join(_ALLOWED_MUNICIPALITIES)
        ))


if USE_DOWNSAMPLING:
    init_downsampling()
