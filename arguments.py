"""
Shared arguments - currently only supports the mighty quietness.

# TODO: add --ground-up or --complete or --reset or --restart or ...? --clean?
#       or --ignore-installed  ~   --ignore-cached  ~
"""
from argparse import ArgumentParser


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


if __name__ == "__main__":
    print(is_quiet())
