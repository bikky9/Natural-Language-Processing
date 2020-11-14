import argparse
from nltk.tree import Tree


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input to the CP2DP transformer")
    parser.add_argument('--DP', dest='DP', help="provide path to the CP txt file")

    args = parser.parse_args()
    DP = args.DP
    DPfile = open(DP, 'r')
