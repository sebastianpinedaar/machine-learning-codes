#!/usr/bin/env python
from operator import itemgetter
import sys

current_key = None
key = None
current_count = 0
current_rating_acc = 0
max_average_rating = 0
max_average_rating_movie = ""
mean_rating = 0

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
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
        if current_key:
            
            # write result to STDOUT
            print (f'ID:{current_key}\tMean:{mean_rating}')
        current_count = count
        current_rating_acc = rating
        current_key = key
        
        if (max_average_rating<mean_rating):
            max_average_rating = mean_rating
            max_average_rating_movie = key
        
# do not forget to output the last key if needed!
if current_key == key:
    print (f'ID:{current_key}\tMean:{mean_rating}')

print("The movie with max. average rating has ID ", max_average_rating_movie, " which is ", max_average_rating)