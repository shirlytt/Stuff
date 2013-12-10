import math
import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
import numpy as np
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


#function that generates a truncated normal using Robert Sampling method
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

# function that generates a trunc-norm by repeatly sampling from normal:
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




def probit_mcmc_cpu(y,X,beta_0,Sigma_0_inv,niter,burnin,verbose=True):
    p = len(beta_0)  # length of regression parameters
    n = len(y)       # number of observations
    nsamples = niter +  burnin
    #Sigma_det = 1/(np.linalg.det(Sigma_0_inv))
    # Calculate counts corresponding to y==0 or y==1
    n0 = len(y[y==0])
    n1 = len(y[y==1])
    # Store the samples:
    gibbs_samples_beta = np.zeros((nsamples,p)).astype(np.float32)
    ## Specify initial values
    beta_curr = np.zeros(p).astype(np.float32)
    z_curr = np.zeros(n).astype(np.float32)
    if (verbose):
        print("\n=============================================\n")
        print("Implementing Gibbs Sampling Algorithm by CPU Code...\n")
        print("=============================================\n\n")
    Sigma_beta = np.linalg.inv(Sigma_0_inv+X.T.dot(X))
    for i in range(nsamples):
        mu_beta = Sigma_beta.dot(Sigma_0_inv.dot(beta_curr)+X.T.dot(z_curr))
        beta_curr = np.random.multivariate_normal(mu_beta,Sigma_beta)
        gibbs_samples_beta[i,:] = beta_curr
        mu_z = X.dot(beta_curr)
        mu_z0,mu_z1 = mu_z[y==0],mu_z[y==1]
        #host = [RS_truncnorm(mu_z0[i],1,-1e6,0) for i in range(n0)]
        #host1 = [RS_truncnorm(mu_z1[i],1,0,1e6) for i in range(n1)]
        #host = [repeat_truncnorm(mu_z0[j],1,-1e6,0) for j in range(n0)]
        #host1 = [repeat_truncnorm(mu_z1[j],1,0,1e6) for j in range(n1)]
        host = [hybrid_truncnorm(mu_z0[i],1,-1e6,0) for i in range(n0)]
        host1 = [hybrid_truncnorm(mu_z1[i],1,0,1e6) for i in range(n1)]
        z_curr[y==0] = host
        z_curr[y==1] = host1
    return gibbs_samples_beta


# --------------------   Implement the CPU code for sampling --------------------- #
#data = np.loadtxt('/home/wwtao/Stuff/HW4/mini_data.txt',delimiter=' ',skiprows=1)
k=4   # k = 1 to 5, deciding the sample to use
#  load the sample, obtain corresponding values:
data = np.loadtxt('data_0'+str(k)+'.txt',delimiter=' ',skiprows=1)
y = data[:,0]
X = data[:,1:]
# set initial values and sampling parameters:
beta_0 = np.zeros(8).astype(np.float32)
Sigma_0_inv = np.zeros((8,8)).astype(np.float32)
niter = 2000
burnin = 500

# start timing the run:
start = time.clock()
smp = probit_mcmc_cpu(y,X,beta_0,Sigma_0_inv,niter,burnin)
elapsed = (time.clock() - start)
sample = smp[burnin:,:]
np.savetxt('data_0'+str(k)+'_CPU_samples.txt',sample,delimiter=' ')


# ==================================================================================






# ----------------      CPU and GPU altogether for sampling --------------------- #
def probit_mcmc_gpu(y,X,beta_0,Sigma_0_inv,niter,burnin,block_dims,grid_dims,
                    verbose=True):
    p = len(beta_0)  # length of regression parameters
    n = np.int32(len(y))       # number of observations
    nsamples = niter +  burnin
    #Sigma_det = 1/(np.linalg.det(Sigma_0_inv))
    # Calculate counts corresponding to y==0 or y==1
    n0 = np.int32(len(y[y==0]))
    n1 = np.int32(len(y[y==1]))
    # Store the samples:
    gibbs_samples_beta = np.zeros((nsamples,p)).astype(np.float32)
    ## Specify initial values
    beta_curr = np.zeros(p).astype(np.float32)
    #beta_curr = np.array([0.568,-0.106,-2.059,0.121,1.053,-0.102,1.233,-0.027]).astype(np.float32)
    if (verbose):
        print("\n=============================================\n")
        print("Implementing Gibbs Sampling Algorithm...\n")
        print("=============================================\n\n")
    Sigma_beta = np.linalg.inv(Sigma_0_inv+X.T.dot(X))
    rng_c_array = np.array(range(nsamples))
    z_curr = np.zeros(n).astype(np.float32)
    vals0 = np.zeros(n0).astype(np.float32)
    vals1 = np.zeros(n1).astype(np.float32)
    rng_a = np.int32(95616)
    rng_b = np.int32(25)
    RS = np.int32(1)
    maxtries = np.int32(2000)
    z_first = np.zeros((15,n)).astype(np.float32)
    for i in range(nsamples):
        mu_beta = Sigma_beta.dot(Sigma_0_inv.dot(beta_curr)+X.T.dot(z_curr))
        beta_curr = np.random.multivariate_normal(mu_beta,Sigma_beta)
        gibbs_samples_beta[i,:] = beta_curr
        mu_z = X.dot(beta_curr).astype(np.float32)
        mu_z0,mu_z1 = mu_z[y==0],mu_z[y==1]
        # specify kernel parameters:
        rng_c = rng_c_array[i]
        # Launch the kernel:
        rtruncnorm(drv.Out(vals0), n0,
              drv.In(mu_z0.astype(np.float32)), drv.In(np.ones(n0).astype(np.float32)),
              drv.In((-1e10)*np.ones(n0).astype(np.float32)),
              drv.In(np.zeros(n0).astype(np.float32)),
              n0, n0,
              n0, n0,
              rng_a, rng_b, np.int32(rng_c),
              maxtries,RS,block=block_dims, grid=grid_dims)
        rtruncnorm(drv.Out(vals1), n1,
              drv.In(mu_z1.astype(np.float32)), drv.In(np.ones(n0).astype(np.float32)),
              drv.In(np.zeros(n1).astype(np.float32)),
              drv.In((1e10)*np.ones(n1).astype(np.float32)),
              n1, n1,
              n1, n1,
              rng_a, rng_b, rng_c,
              maxtries,RS,block= block_dims, grid= grid_dims)
        z_curr[y==0] = vals0
        z_curr[y==1] = vals1
        if i<15:
            z_first[i,:]=z_curr
    return gibbs_samples_beta
    #return z_first


# --------------------   Choose the sample size and corresponding block,grid sizes --------------------- #
k=1
grid_dims = (1,1)


# or
k=2
grid_dims = (16,16,1)

# or
k=3
grid_dims = (391,1,1)

# or
k=4
grid_dims = (3907,1,1)

# or
k=5
grid_dims = (39063,1,1)


# and
block_dims = (16,16,1)


# --------------------   Implement the GPU code for sampling --------------------- #
#data = np.loadtxt('mini_data.txt',delimiter=' ',skiprows=1)
data = np.loadtxt('data_0'+str(k)+'.txt',delimiter=' ',skiprows=1)
y = data[:,0]
X = data[:,1:]
beta_0 = np.zeros(8).astype(np.float32)
#beta_0 = np.array([0.568,-0.106,-2.059,0.121,1.053,-0.102,1.233,-0.027]).astype(np.float32)
Sigma_0_inv = np.zeros((8,8)).astype(np.float32)
niter = 2000
burnin = 500



start = time.clock()
temp = probit_mcmc_gpu(y,X,beta_0,Sigma_0_inv,niter,burnin,block_dims,grid_dims)
elapsed = (time.clock() - start)  # k=1:  8.42s; k=2: 10.09; k=3: 135.04; k=4: 1395;
sample = temp[burnin:,:]
np.savetxt('data_0'+str(k)+'_GPU_samples.txt',sample,delimiter=' ')

