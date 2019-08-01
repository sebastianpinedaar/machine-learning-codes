#!/usr/bin/env python
import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.split(",")

    genders = line[6]
    rating = line[3]
    for g in genders.split("|"):
        g = g.strip()
        print (f"{g}\t{rating}\t1")

