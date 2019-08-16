"""
Shared arguments - currently only supports the mighty quietness.

# TODO: add --ground-up or --complete or --reset or --restart or ...? --clean?
#       or --ignore-installed  ~   --ignore-cached  ~
"""
from argparse import ArgumentParser


def is_quiet():
    parser = ArgumentParser()
    parser.add_argument("--quiet",  action="store_true")

    args = parser.parse_args()
    return args.quiet


if __name__ == "__main__":
    print(is_quiet())
