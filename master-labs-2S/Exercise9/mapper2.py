#!/usr/bin/env python
import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip().strip("\"")
    # split the line into words
    fields = line.split(",")
    # increase counters
 
    airport = fields[4]
    delay = fields[8]

    if (len(delay)==0):
        delay=0

    print (f"{airport}\t{delay}\t1")
   
       
       
