#!/usr/bin/env python
import sys
import numpy as np

current_key = None
key = None
current_count = 0
current_delay_acc = 0
mean_delay = 0
current_mean_list = []
current_airline_list = []

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
        mean_delay = round(current_delay_acc/current_count,2)
        
    else:
        if current_key:
            # write result to STDOUT
            print (f'ID:{current_key}\tMean:{mean_delay}')
            
            mean_delay = round(current_delay_acc/current_count,2)       
            current_mean_list.append(mean_delay)
            current_airline_list.append(key)
        
        current_count = count
        current_delay_acc = delay
        current_key = key


if current_key == key:
    current_mean_list.append(mean_delay)
    current_airline_list.append(key) 
    
idx = np.array(np.argsort(-np.array(current_mean_list)))[:11]
out = dict(zip(np.array(current_airline_list)[idx], np.array(current_mean_list)[idx]))

print (out)