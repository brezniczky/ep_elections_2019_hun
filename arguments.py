"""
Shared arguments - currently only supports the mighty quietness.
"""
from argparse import ArgumentParser


def is_quiet():
    parser = ArgumentParser()
    parser.add_argument("--quiet",  action="store_true")

    args = parser.parse_args()
    return args.quiet


if __name__ == "__main__":
    print(is_quiet())
