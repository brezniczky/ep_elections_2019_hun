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
DEFAULT_QUICK_RUN_OUTPUT_DIRECTORY = "quick_output"
_DEFAULT_OUTPUT_DIRECTORY_POSTFIXES = []


def get_parsed_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--quiet", action="store_true",
        help="run without popping up blocking dialogs (e.g. for "
             "charts) and only printing a brief ouptut"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="run with a reduced workload - a quick integration testing"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="place output files into this directory instead of "
              "the default (%s)" % DEFAULT_OUTPUT_DIRECTORY
    )
    args = parser.parse_known_args()[0]
    return args


def is_quiet():
    args = get_parsed_args()
    # we're conflicting with whatever's around us (often Jupyter)
    if "quiet" in args:
        return args.quiet
    # ugly - quick workaround to be able to re-generate the final ipynb ==> HTML
    # this fallback case is to be expected to take place from within Jupyter NB
    return False


def is_quick(args=None):
    if args is None:
        args = get_parsed_args()
    return args.quick if "quick" in args else False


def get_default_output_dir(args=None):
    ans = (
        DEFAULT_OUTPUT_DIRECTORY
        if not is_quick(args)
        else DEFAULT_QUICK_RUN_OUTPUT_DIRECTORY
    )
    for postfix in _DEFAULT_OUTPUT_DIRECTORY_POSTFIXES:
        ans += postfix
    return ans


def add_output_dir_postfix(postfix):
    global _DEFAULT_OUTPUT_DIRECTORY_POSTFIXES
    _DEFAULT_OUTPUT_DIRECTORY_POSTFIXES.append(postfix)


def get_output_dir():
    args = get_parsed_args()
    ans = (args.output_dir
           if "output_dir" in args and args.output_dir is not None
           else get_default_output_dir(args))
    if not os.path.exists(ans):
        os.mkdir(ans)
    return ans


def output_exists(filename: str) -> bool:
    return os.path.exists(os.path.join(get_output_dir(), filename))


def save_output(df: pd.DataFrame, filename: str):
    filename = os.path.join(get_output_dir(), filename)
    # someone wrote explicit (albeit otherwise default) utf-8 spec. is a
    # good practice - who am I to argue (until I properly thought this
    # through ...)
    df.to_csv(filename, index=False, encoding="utf8")


def load_output(filename: str) -> pd.DataFrame:
    filename = os.path.join(get_output_dir(), filename)
    df = pd.read_csv(filename)
    return df


if __name__ == "__main__":
    print(is_quiet())
    print(is_quick())
    print(get_output_dir())
