#!/usr/bin/env python
import sys

current_key = None
key = None
current_count = 0
current_rating_acc = 0
current_max = 0
mean_rating = 0 
    
# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    print(line)
    line = line.strip()
    
    # parse the input we got from mapper.py
    key, value = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
       rating, count = value.split('\t')
       count = int(count)
       rating = float(rating)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key  before it is passed to the reducer
    if current_key == key:
        current_count += count
        current_rating_acc += rating
        mean_rating = round(current_rating_acc/current_count,2)
        
    else:

        if(current_max< mean_rating):         
            current_max = mean_rating
            genre_max = key
                
        current_count = count
        current_rating_acc = rating
        current_key = key

print (f'Genre:{genre_max}\tMean:{current_max}')