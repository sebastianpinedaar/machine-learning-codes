#!/usr/bin/env python
import sys

#initializing values
current_key = None
key = None
current_count = 0
current_rating_acc = 0
current_min = 5
min_key = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper
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
        if current_key:
            # write result to STDOUT
            if (current_count>40):
                print (f'ID:{current_key}\tMean:{mean_rating}\tCount:{current_count}')
                
                if(current_min> mean_rating):
                    current_min = mean_rating
                    min_key = current_key
                    
        current_count = count
        current_rating_acc = rating
        current_key = key

if current_key == key and current_count>40:
    print (f'ID:{current_key}\tMean:{mean_rating}')
    
print("The user ID %s has the lowest rating mean (%s)" % (min_key, current_min))
