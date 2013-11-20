__author__ = 'shirlytao'

import sys
import csv
import numpy as np
import wls
import read_some_lines as rsl

print
"=============================="
print
"Python version:"
print
str(sys.version)
print
"Numpy version:"
print
str(np.version.version)
print
"=============================="

mini = False
verbose = False

datadir = "/home/pdbaines/data/"
outpath = "output/"

# mini or full?
if mini:
    rootfilename = "blb_lin_reg_mini"
else:
    rootfilename = "blb_lin_reg_data"

########################################################################################
## Handle batch job arguments:

nargs = len(sys.argv)
print
'Command line arguments: ' + str(sys.argv)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start = 1000
length_datasets = 250    # composed of s = 5 subsets and r = 50 bootstrap samples for each subset
r = 50
s = 5
#######################

# Note: this only sets the random seed for numpy, so if you intend
# on using other modules to generate random numbers, then be sure
# to set the appropriate RNG here

if (nargs == 1):
    sim_num = sim_start + 1
    sim_seed = 1330931
else:
    # Decide on the job number, usually start at 1000:
    sim_num = sim_start + int(sys.argv[nargs - 1])
    # Set a different random seed for every job number!!!
    sim_seed = 762 * sim_num + 1330931

# Bootstrap datasets numbered 1001-1250

########################################################################################
########################################################################################

# Find r and s indices:
s_index = (sim_num - sim_start) // r + 1
r_index = (sim_num - sim_start) % r
if r_index == 0:
    r_index = 50
    s_index = s_index - 1

print
"========================="
print
"sim_num = " + str(sim_num)
print
"s_index = " + str(s_index)
print
"r_index = " + str(r_index)
print
"========================="

# filename:
filename = str(datadir + rootfilename+'.txt')
#filename = str( rootfilename+'.txt')

# dataset specs: parameters involved in the bags-of-bootstrap
gamma = 0.7
n = 1000000  # number of rows
nc = 1001     # number of columns
b = int(n ** gamma)     # the subset size, shared by all s subsets.
if mini:
    n = 10000
    nc = 41
    b = int(n ** gamma)     # the subset size, shared by all s subsets.


# Set random seed so indices are same for each s:
sim_seed_s = 762 * (s_index + sim_start) + 1330931   # the simulation seed for generating indices of a subset

# sample indices:  generate the b indices uniformly  selected from (1: data_size) without replacement
np.random.seed(sim_seed_s)   # use the same simulation seed to generate the same subset
indices = np.random.choice(n, b,replace=False)    # these indices should be shared by all b jobs such that they run on the same subset
indices = np.sort(indices)    # sort the indices
nr = len(indices)

# Reset random seed so things differ:
np.random.seed(sim_seed)

# Take the subset: function read_some_lines_csv is from module rsl and reads lines in the full dataset with function
# read_some_lines_csv
subset = rsl.read_some_lines_csv(filename, indices, nr, nc, n, print_every=1000, verbose=False)

# Bootstrap to full datasize:
if mini:
    indices_b = np.random.multinomial(n, [1 / 630.] * 630)
else:
    indices_b = np.random.multinomial(n, [1 / 15848.] * 15848)

# fit linear model with weights to the bootstrap dataset
lr_fit = wls.wls(subset[:, nc - 1], subset[:, :(nc - 1)], indices_b[:])


# Output file:
outfile = "output/mini_coef_%02d_%02d.txt" %(s_index,r_index)
# Write to file:
#with open(outfile, 'w') as fout:
 #  fout.write(str(lr_fit)) #it is the first column
np.savetxt(outfile, lr_fit)