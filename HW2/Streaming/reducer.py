#!/usr/bin/env python
import sys

current_bin   = None
current_count = 0
bin = None


for line in sys.stdin:
    line = line.strip()
    bin, count  = line.split('\t',1) # 1 is the maximum number of split
    #bin_x,bin_y = bin.split(',') # bin_x and bin_y contain the string format of the lower bound of the bin along x,y
    try:
        bin_x,bin_y = bin.split(',')
        count = int(count)
    except ValueError:
        continue
    if current_bin == bin:
        current_count += count                                   # increase the count for current bin by 1
    else:                                               # else, if the current line does not match current bin...
        if current_bin:
            # write result to stdout
            #print current_bin
            current_bin_x,current_bin_y = current_bin.split(',')
            bin_key = current_bin_x + ','+str(float(current_bin_x)+0.1) +str(',')+ current_bin_y +','+ str(float(current_bin_y)+0.1)+str(',')
            print '%s\t%s' % (bin_key,current_count) # current bin is full, print its count
        # since current line does not match current bin, set current bin to next bin and set count = 1
        current_count   = count
        current_bin     = bin
        #current_bin_x_hi = float(current_bin_x)+0.1   # upper bound of the bin along x-axis
        #current_bin_y_hi = float(current_bin_y)+0.1   # upper bound of the bin along y-axis

if current_bin == bin:
    current_bin_x,current_bin_y = current_bin.split(',')
    bin_key = current_bin_x + str(',')+str(float(current_bin_x)+0.1) +str(',')+ current_bin_y +','+ str(float(current_bin_y)+0.1)+str(',')
    print '%s\t%s' % (bin_key,current_count) # current bin is full, print its count
