#!/usr/bin/env python3

import os.path
import sys
import yaml
import pykwalify.core
import logging
logging.disable(logging.CRITICAL)


def main():
    args = sys.argv[1:]

    if len(args) != 2:
        print("usage: {} <YAML_FILE> <YAML_SCHEMA_FILE>".format(sys.argv[1:]))
        print("       pip3 install pykwalify")

        exit(2)
    # end if

    if not os.path.isfile(args[0]):
        print("Parameter YAML_FILE ({}) is no file.".format(args[0]))
        exit(1)
    # end if

    if not os.path.isfile(args[1]):
        print("Parameter YAML_SCHEMA_FILE ({}) is no file.".format(args[1]))
        exit(1)
    # end if

    try:
        c = pykwalify.core.Core(source_file=args[0], schema_files=[args[1]])

        try:
            c.validate(raise_exception=True)

        except pykwalify.core.SchemaError as e:
            print(e)
            exit(1)
        # end try

    except BaseException as e:
        print(e)
        exit(1)
    # end try

    exit(0)  # Success
# end def main


if __name__ == "__main__":
    main()
