#!/usr/bin/env python
import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.split(",")
    
    if(line[0].isalpha()):
        pass
    else: 
        movie = line[1]
        rating = float(line[2])
        print (f"{movie}\t{rating}\t1")
   