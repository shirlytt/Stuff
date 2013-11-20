#!/usr/bin/env python
import sys
for line in sys.stdin:
    # remove leading and trailing whitespace
    line  = line.strip()
    # split the line into numbers
    po_bound0,po_bound1 = line.split('\t')
    po_bin_bound_x   = float(int(float(po_bound0)*10)*0.1) # convert decimal into a.b format,
                                                     # which will be the lower bound of the bin it is in (x-axis)
    #po_bin_bound_y   = float(int(float(po_bound1)*10))/10
    po_bin_bound_y    = float(int(float(po_bound1)*10)*0.1)
    # write results to STDOUT
    bin_key = str(po_bin_bound_x)+str(',')+str(po_bin_bound_y)
    key = bin_key.strip()
    print '%s\t%s' % (key,1)

