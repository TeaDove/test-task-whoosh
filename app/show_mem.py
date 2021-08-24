#!/usr/bin/python3

import sys

import numpy as np

if __name__ == "__main__":
    if sys.argv[1] == "full":
        for filename in sys.argv[2:]:
            fp = np.memmap(filename, 'int32', 'r')
            for item in fp[::1_000]:
                print(item, end=',')
    else:
        print("FILENAME\tDATA\t\t\t\t\t\t\t\t\tMIN\tMAX\t\tSIZE")
        for filename in sys.argv[1:]:
            fp = np.memmap(filename, 'int32', 'r')
            print("{}:\t{}\t{:,}\t{:,}\t{:,}".format(filename, fp, fp.min(), fp.max(), fp.shape[0]))
