"""
Shared arguments - currently only supports the mighty quietness.

# TODO: add --ground-up or --complete or --reset or --restart or ...? --clean?
#       or --ignore-installed  ~   --ignore-cached  ~
"""
"""
TODO: make this os_utils. could be envrionment variables, etc. and otherwise 
      it's just too small
"""
from argparse import ArgumentParser
import os
import pandas as pd


DEFAULT_OUTPUT_DIRECTORY = "output"


def is_quiet():
    parser = ArgumentParser()
    parser.add_argument("--quiet", action="store_true")

    # we're conflicting with whatever's around us (often Jupyter)
    args = parser.parse_known_args()[0]
    if "quiet" in args:
        return args.quiet
    # ugly - quick workaround to be able to re-generate the final ipynb ==> HTML
    # this fallback case is to be expected to take place from within Jupyter NB
    return False


def get_output_dir():
    ans = DEFAULT_OUTPUT_DIRECTORY
    if not os.path.exists(ans):
        os.mkdir(ans)
    return ans


def output_exists(filename: str) -> bool:
    return os.path.exists(os.path.join(get_output_dir(), filename))


def save_output(df: pd.DataFrame, filename: str):
    filename = os.path.join(get_output_dir(), filename)
    # someone wrote explicit (albeit otherwise) utf-8 spec. is a good practice
    # - who am I to argue (until I properly thought this through ...)
    df.to_csv(filename, index=False, encoding="utf8")


def load_output(filename: str) -> pd.DataFrame:
    filename = os.path.join(get_output_dir(), filename)
    df = pd.read_csv(filename)
    return df


if __name__ == "__main__":
    print(is_quiet())
