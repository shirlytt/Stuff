
##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

library(mvtnorm)
library(coda)

########################################################################################
########################################################################################
## Handle batch job arguments:

# 1-indexed version is used now.
args <- commandArgs(TRUE)

cat(paste0("Command-line arguments:\n"))
print(args)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start <- 1000
length.datasets <- 200
#######################

if (length(args)==0){
  sinkit <- FALSE
  sim_num <- sim_start + 1
  set.seed(1330931)
} else {
  # Sink output to file?
  sinkit <- TRUE
  # Decide on the job number, usually start at 1000:
  sim_num <- sim_start + as.numeric(args[1])
  # Set a different random seed for every job number!!!
  set.seed(762*sim_num + 1330931)
}

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################
log.target.density <- function(beta,m,y,X,beta.0,S.inv0=diag(2),S.det  = 1){
    # X is of dim n*2
   n <- length(y)
   p <- length(beta.0)
   beta = beta - beta.0
    log.density = - p/2*log(2*pi) - 0.5*t(beta)%*%S.inv0%*%beta  - 0.5*log(S.det) +
                 sum(apply(cbind(X,y),1,function(x){x[1:p]%*%beta*x[length(x)]})) -
                 sum(m*apply(X,1,function(x){log(1+exp(x%*%beta))})) +
                 sum(apply(cbind(m,y),1,function(my){log(factorial(my[1])/(factorial(my[2])*factorial(my[1]-my[2])))}))
   return(log.density)
}


bayes.logreg <- function(n,y,X,beta.0,Sigma.0.inv,niter=10000,burin=1000,
                           print.every=1000,retune=100,verbose=TRUE)
{
	total.samples <- niter +  burin
    # Store the samples:
    gibbs.samples <- matrix(NA,nrow=total.samples,ncol=length(beta.0))
   
    n <- length(y)
    p <- length(beta.0)
    S.det = 1/(det(Sigma.0.inv))
    ## Specify initial values
     # Starting state:
     beta1.curr <- 0
     beta0.curr <- 0
     # Track tha acceptance rate:
       n.accept <- rep(0,p)
     # Proposal distribution: univariate normal -- N(beta0.prop-beta0.curr, v^2) [jumping step]
     v <- 1
     v2 <- 1.18
     rho <- 0.024
    # within each Gibbs sampler, implement MH

           ######################################################
           # MCMC using Metropolis/Metropolis-Hastings Algorithm
           ######################################################
 
    do.metropolis <- TRUE
    do.MH <- FALSE
    
    if (verbose){
       cat("\n=============================================\n")
       cat("Implementing Metropolis Algorithm...\n")
       cat("=============================================\n\n")
      } 
    # Metropolis-Markov Chains Algorithm within Gibbs samplers:
   if (do.metropolis){
    for (i in 1:total.samples){
      # Propose a new state:
      beta0.prop <- rnorm(n=1,mean=beta0.curr,sd=v)
      # Compute the log-acceptance probability: 
      log.density.prop = log.target.density(beta=c(beta0.prop,beta1.curr),m=m,y=y,X=X,beta.0=beta.0,S.inv0=Sigma.0.inv,S.det  = S.det)
      log.density.curr = log.target.density(beta=c(beta0.curr,beta1.curr),m=m,y=y,X=X,beta.0=beta.0,S.inv0=Sigma.0.inv,S.det = S.det)
      log.alpha <-  log.density.prop  - log.density.curr
      # Decide whether to accept or reject:
      log.u <- log(runif(1))
      if (log.u < log.alpha){
        # Accept:
        beta0.curr <- beta0.prop
        n.accept[1] <- n.accept[1] + 1
      } else {
        # Reject
      }
  
      # similarly for beta1
       beta1.prop <- rnorm(n=1,mean=beta1.curr,sd=v)
      # Compute the log-acceptance probability: log(alpha) = log( pi(theta^prop)/pi(theta^curr) )
      log.density.prop = log.target.density(beta=c(beta0.curr,beta1.prop),m=m,y=y,X=X,beta.0,S.inv0=Sigma.0.inv,S.det  = S.det)
      log.density.curr = log.target.density(beta=c(beta0.curr,beta1.curr),m=m,y=y,X=X,beta.0,S.inv0=Sigma.0.inv,S.det  = S.det)
      log.alpha <-  log.density.prop-log.density.curr
      # Decide whether to accept or reject:
      log.u <- log(runif(1))
      if (log.u < log.alpha){
        # Accept:
        beta1.curr <- beta1.prop
        n.accept[2] <- n.accept[2] + 1
      } else {
        # Reject
      }
  
      # Store the current state:
      gibbs.samples[i,1] <- beta0.curr
      gibbs.samples[i,2] <- beta1.curr
      
      
      # Check acceptance rate and whether to tune v
       if (i%% retune == 0  & i<burin){
           if(any(n.accept[1]/i<0.3, n.accept[2]/i<0.3)){
              v = v / 2
            } 
           if (any(n.accept[1]/i>0.6, n.accept[2]/i>0.6)){
              v = v * 2  
            }
          if (verbose & any(n.accept[1]/i<0.3, n.accept[2]/i<0.3,n.accept[1]/i>0.6, n.accept[2]/i>0.6)){
           cat(paste('This is iteration ',i,', current acceptance rate is ', round(100*n.accept[1]/i)/100,' and ',round(100*n.accept[2]/i)/100, ', change v to ',v,'\n',sep=''))
          }
         } 
      # print an update to the user
      if (i%%print.every == 0 & verbose){
      cat(paste("This is iteration",i,"and the new sampled beta is",
          gibbs.samples[i,1],gibbs.samples[i,2],"\n",sep=" "))
      }
     }
   return(list(n.accept=n.accept, sample = gibbs.samples[(burin+1):total.samples,]))
   }
   
   # Metropolis-Markov Chains Algorithm by itself
   if (do.MH){
      library(mvtnorm)
      Sigma.prop = v* matrix(c(1,rho*sqrt(v2),rho*sqrt(v2),v2),nrow=2)*0.01
      beta.curr = c(beta0.curr,beta1.curr)
      for (i in 1:total.samples){
        # Propose a new state:
        beta.prop <- rmvnorm(n=1,mean=beta.curr,sigma = Sigma.prop)
        beta.prop = t(beta.prop)
        # Compute the log-acceptance probability: log(alpha) = log( pi(theta^prop)/pi(theta^curr) )
        log.alpha <- log.target.density(beta.prop,m=m,y=y,X=X,beta.0=beta.0,S.inv0=Sigma.0.inv,S.det  = S.det) - 
                  log.target.density(beta.curr,m=m,y=y,X=X,beta.0=beta.0,S.inv0=Sigma.0.inv,S.det = S.det)
        # Decide whether to accept or reject:
        log.u <- log(runif(1))
        if (log.u < log.alpha){
          # Accept:
          beta.curr <- beta.prop
          n.accept[1] <- n.accept[1] + 1
         } else {
          # Reject
         }
        # Store the current state:
        gibbs.samples[i,] <- beta.curr
        # Check acceptance rate and whether to tune v
         if (i%% retune == 0  & i<burin){
           if(any(n.accept[1]/i<0.3, n.accept[2]/i<0.3)){
              v = v / 2
            } 
           if (any(n.accept[1]/i>0.6, n.accept[2]/i>0.6)){
              v = v * 2  
            }
          if (verbose & any(n.accept[1]/i<0.3, n.accept[2]/i<0.3,n.accept[1]/i>0.6, n.accept[2]/i>0.6)){
           cat(paste('This is iteration ',i,', current acceptance rate is ', round(100*n.accept[1]/i)/100,' and ',round(100*n.accept[2]/i)/100, ', change v to ',v,'\n',sep=''))
          }
         }  
      # print an update to the user
      if (i%%print.every == 0 & verbose ){
      cat(paste("This is iteration",i,"and the new sampled beta is [",
          gibbs.samples[i,1],gibbs.samples[i,2],"]\n",sep=" "))
      }
     }
   return(list(n.accept=n.accept, sample = gibbs.samples[(burin+1):total.samples,]))
   }
}

#################################################
# Set up the specifications:
p     <-  2
beta.0 <- matrix(c(0,0))
Sigma.0.inv <- diag(rep(1.0,p))
niter <- 10000
burin <- 1000
print.every  <- 1000
retune      <-  100
verbose <- TRUE

# etc... (more needed here)
#################################################

# Read data corresponding to appropriate sim_num:
data = read.csv(paste('./data/blr_data_',sim_num,'.csv',sep=''),header=TRUE)
true_beta = read.csv(paste('./data/blr_pars_',sim_num,'.csv',sep=''),header=TRUE)

# Extract X and y:
m = data$n
y = data$y
X = cbind(data$X1,data$X2)

# Fit the Bayesian model:
result = bayes.logreg(m,y,X,beta.0,Sigma.0.inv,niter,burin,print.every,retune,verbose)

# Extract posterior quantiles...
gibbs.quantile = cbind(as.vector(100*quantile(result$sample[,1],(1:99)/100)/100), 
                      as.vector(100*quantile(result$sample[,2],(1:99)/100)/100))
                      
# Write results to a (99 x p) csv file...

write.table(data.frame(gibbs.quantile) ,file=paste('./results/blr_res_',sim_num,'.csv',sep=''),sep=',',row.names=FALSE,col.names=FALSE)

# Go celebrate.
 
cat("done. :)\n")








