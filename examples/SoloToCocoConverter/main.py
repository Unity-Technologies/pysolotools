# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from Converter import *
import os


def preview(args):
    args = [args.data]
    solo_to_panda_converter(args)

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--data', type=str,
                 help='path to dataset', default="")
    preview(cli.parse_args())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
