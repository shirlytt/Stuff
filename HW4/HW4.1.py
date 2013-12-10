# import required modules:
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
import numpy as np
import math
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>
    #include <stdlib.h>
    #include <curand_kernel.h>
    #include <math_constants.h>
    #include <math.h>

    extern "C"
    {
    //   TRUNCNORM_THREAD_CNT ???

    __global__ void
    rtruncnorm_kernel(float *vals, int n,
                      float *mu, float *sigma,
                      float *lo, float *hi,
                      int mu_len, int sigma_len,
                      int lo_len, int hi_len,
                      int rng_a, int rng_b, int rng_c,
                      int maxtries)
    {

        // Usual block/thread indexing...
        int myblock = blockIdx.x + blockIdx.y * gridDim.x;
        int blocksize = blockDim.x * blockDim.y * blockDim.z;
        int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
        int idx = myblock * blocksize + subthread;

        if (idx < n){
         // Setup the RNG:
         curandState rng;   /* define rng to be of class curandState... */
         /* initialize the random state for current thread */
         curand_init(rng_a + idx*rng_b,rng_c,0,&rng);
         /* parameters above -- The first is seed:  an integer that initializes the starting state of a pseudorandom
                             number generator. The same seed always produces the same sequence of results. */
         /*                  --  The second offset parameter is used to skip ahead in the sequence.
                             If offset = 100, the first random number generated will be the 100th
                             in the sequence. This allows multiple runs of the same program to continue
                             generating results from the same sequence without overlap */
         /*                  -- The third parameter in curand_init is set as 0, meaning that giving each thread
                             a different seed and just keep the sequence number at 0 for every thread */
         // Draw sample:
         int accepted = 0;
         int numtries = 0;
         /* Draw two-sided truncated normal sample by repeat normal sampling */
         float x = 0;
         while (!accepted && numtries < maxtries){
           numtries ++;
           x = mu[idx] +  curand_normal(&rng)*sigma[idx];
           if (x > lo[idx] && x < hi[idx]){
                accepted = 10;
                break;
           }
         }
         if (numtries == maxtries){
             int numtries1 = 0;
             float rho  = 0;
             float z    = 0;
             while (!accepted && numtries1 < maxtries){
              numtries1 ++;
              // generate a uniformly distributed rv on [0,1]:
              float u   =  curand_uniform(&rng);
              if (hi[idx]<mu[idx]){
                 // this indicates  a right-truncation situation
                 float hi0 =  mu[idx] - hi[idx]; // a positive value, transform into left-trunc
                 float alpha = (hi0 + sqrt(hi0*hi0+4)) / 2;
                 // generate r.v. z with distribution as translated exponential:
                 z   =   log(1-u)/alpha - hi0; // a negative value
                 rho =  expf(-(alpha - hi0)*(alpha-hi0)/2);
                 }
              else if (lo[idx]>mu[idx]){
                 // this indicates  a left-truncation situation
                  float lo0 = lo[idx] - mu[idx];
                  float alpha = (lo0 + sqrt(lo0*lo0 + 4))/2;
                 // generate r.v. z with distribution as translated exponential:
                  z   =  - log(1-u)/alpha + lo0;
                  rho =  expf(-(alpha - lo0)*(alpha-lo0)/2);
                  }
              else {
                  z   =  curand_uniform(&rng)*(hi[idx]-lo[idx])+lo[idx]-mu[idx];
                  rho =  expf(-z*z/2);
                  }
              float v   =  curand_uniform(&rng);
              if (v < rho) {
                x = z + mu[idx];
                accepted = 10;
                break;
               }
            } // finish current while-loop
           }  // finishes numtries == maxtries if-statement
        // Store sample:
         vals[idx]= x;
        }
        return;
    }  // END definition of function rtruncnorm_kernel

    } // END extern "C"
    """, include_dirs=['/usr/local/cuda/include/'], no_extern_c=1)


# obtain the kernel function
rtruncnorm = mod.get_function("rtruncnorm_kernel")



# Arguments must be numpy datatypes i.e., n = 1000 will not work!
n = np.int32(1e7)  # sample size, from 1e1 to 1e8

# Threads per block and number of blocks:
# For simplicity in Q1 we use 1d blocks and grids.
tpb = int(512)
nb = int(1 + (n / tpb))

# input parameters: some of them need to be  vectorized:
mu = 2.0 * np.ones(n).astype(np.float32)
sigma = 1.0 * np.ones(n).astype(np.float32)
lo = -5 * np.ones(n).astype(np.float32)
hi = -3 * np.ones(n).astype(np.float32)
mu_len, sigma_len, lo_len, hi_len = n, n, n, n

# Allocate storage for the result:
vals = np.zeros(n).astype(np.float32)

# Set the function input parameters: rng_a,rng_b,rng_c set random seeds
rng_a = np.int32(95616)
rng_b = np.int32(25)
rng_c = np.int32(250)
maxtries = np.int32(2000)



# Create two timers:
start = drv.Event()
end = drv.Event()

start.record()
# Launch the kernel:
rtruncnorm(drv.Out(vals), n,
              drv.In(mu), drv.In(sigma),
              drv.In(lo), drv.In(hi),
              mu_len, sigma_len,
              lo_len, hi_len,
              rng_a, rng_b, rng_c,
              maxtries, block=(tpb, 1, 1), grid=(nb, 1))
end.record()
end.synchronize()
gpu_secs = start.time_till(end) * 1e-3
print("SourceModule time: %f" % gpu_secs)
print("\n")


# Function used for pure-CPU case, to draw 2-sided truncated normal sample:
def RS_truncnorm(mu,sd,a,b,maxtries=2000,type=2,verbose=False):
    numtries = np.int32(0)
    accepted = False
    sample = np.float32(0.0)
    while ((not accepted) and numtries < maxtries):
        numtries += 1
        z = np.random.uniform(a-mu,b-mu,1)
        if (a-mu<0 and b-mu>0):
            rho = math.exp(-z**2/2)
        elif b-mu<0:
            rho = math.exp(((b-mu)**2-z**2)/2)
        else:
            rho = math.exp(((a-mu)**2-z**2)/2)
        u = np.random.uniform(0,1,1)
        if (u<rho):
            sample = z + 2
            accepted = True
            break
    return sample

def repeat_truncnorm(mu,sd,a,b,maxtries=2000,type=2):
    numtries = np.int32(0)
    x = 0
    while(numtries<maxtries):
        numtries += 1
        z=np.random.normal(size=1,loc=mu,scale=sd)
        if (z>a and z<b):
            x = z
            break
    return x

def hybrid_truncnorm(mu,sd,a,b,maxtries=2000,type=2):
    numtries = np.int32(0)
    x = 0
    while(numtries<maxtries):
        numtries += 1
        z = np.random.normal(size=1,loc=mu,scale=sd)
        if (z>a and z<b):
            x = z
            break
    if numtries == maxtries:
        x = RS_truncnorm(mu,sd,a,b,maxtries,type)
    return x


#  ---------------------------  CPU method 1     ---------------------------- #
start.record()
host = np.random.normal(size=n * 1e2, loc=2.0, scale=1.0)  # draw many more samples
# select those falling in the truncated interval
samples = [x for x in host if (x > 0.5 and x < 1.5)]
if np.shape(samples)[0] > n:
    vals_numpy = samples[:n]
else:
     # use the satisfied ones from repeated normal
    m =  np.shape(samples)[0]
    vals_numpy[:m] = samples[:n]
    vals_numpy[m:] = [RS_truncnorm(mu[0],sigma[0],lo[0],hi[0],
                                   maxtries) for i in range(n-m)]

end.record() # end timing
# calculate the run length
end.synchronize()
cpu_secs = start.time_till(end) * 1e-3
print("       Numpy time: %fs" % cpu_secs)
print("\n")
print("Numpy -- expectation of the sample  is: %f" %np.average(vals_numpy))
print("\n")

#  ---------------------------  CPU method 1     ---------------------------- #
start.record()
vals_numpy  = [hybrid_truncnorm(mu[0],sigma[0],lo[0],hi[0],
                                   maxtries) for i in range(n)]

end.record() # end timing
# calculate the run length
end.synchronize()
cpu_secs = start.time_till(end) * 1e-3
print("       Numpy time: %fs" % cpu_secs)
print("\n")
print("Numpy -- expectation of the sample  is: %f" %np.average(vals_numpy))
print("\n")


