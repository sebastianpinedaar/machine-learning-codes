#!/usr/bin/env python
import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.split(",")
    
    if(line[0].isalpha()):
        pass
    else: 
        user = line[0]
        rating = float(line[2])
        print (f"{user}\t{rating}\t1")