#!/usr/bin/env python

import sys

current_key = None
key = None
current_count = 0
current_delay_acc = 0
current_max = 0
current_min = 100000
        

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    key, value = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
       delay, count = value.split('\t')
       count = int(count)
       delay = float(delay)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key before it is passed to the reducer
    if current_key == key:
        current_count += count
        current_delay_acc += delay
        current_max = max(current_max, delay)
        current_min = min (current_min, delay)
        mean_delay = round(current_delay_acc/current_count,2)
        
    else:
        if current_key:
            # write result to STDOUT
            print (f'ID:{current_key}\tMean:{mean_delay}\tMin:{current_min}\tMax:{current_max}')
        current_count = count
        current_delay_acc = delay
        current_min = delay
        current_max = delay
        current_key = key


if current_key == key:
    print (f'ID:{current_key}\tMean:{mean_delay}\tMin:{current_min}\tMax:{current_max}')