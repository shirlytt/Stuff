######   STA250 HW3
######   Wenwen Tao
######   11-24-2013


###################################################################################
#####  Problem 1a 
#####  Implement bisection algorithm
Bisection = function(f, l, u, conv.cri, iter, verbose=FALSE){
  converged = FALSE
  step=1
  while(!converged & step<iter){
  midpoint = (l+u)/2
  if(abs(f(midpoint))<conv.cri){
    converged = TRUE
  }else{
    if(f(l)*f(midpoint)<0){
      u = midpoint
    }else if(f(u)*f(midpoint)<0){
      l = midpoint 
    }
  }
  if(verbose==TRUE){
    cat("iteration step:",step,", next interval is: (",l,",",u,")","\n",sep="")
  }
  step = step + 1
  }
  if(verbose==TRUE){
    cat("iteration step:",step,", root is: ",midpoint,"\n",sep="")
  }
  return(midpoint)
}

f = function(a){
  return(2*a+1)
}

Bisection(f, -99, 3, 0.000001, iter=100, verbose=TRUE)

###################################################################################
#####  Problem 1b 
#####  Implement Newton-Raphson algorithm

NR = function(f, g, ini, conv.cri, iter, verbose){
  converged = FALSE
  step=1
  root.old = ini
  while(!converged & step<iter){
    if(abs(f(root.old)) < conv.cri){
      converged = TRUE
    }else{
      root.new = root.old - f(root.old)/g(root.old)
      root.old = root.new
    }
  if(verbose==TRUE){
      cat("iteration step:",step,", function value is:",f(root.old),"\n",sep="")
   }
  step = step + 1  
  }
  if(verbose==TRUE){
    cat("iteration step:",step,", root is:",root.old,"\n",sep="")
  }
  return(root.old)
}


f = function(a){
  return(log(a))
}

g = function(a){
  return(1/a)
}


NR(f, g, ini=0.000001, conv.cri=0.00001, iter=100, verbose=TRUE)


###################################################################################
#####  Problem 1c 
#####  Implement Bisection and Newton-Raphson algorithm to linkage function

f = function(a){
  return(125/(2+a) - 38/(1-a) + 34/a)
}

g = function(a){
  return(-125/(2+a)^2 - 38/(1-a)^2 - 34/a^2)
}

Bisection(f, 0, 1, 0.000001, iter=100, verbose=TRUE)
NR(f, g, ini=0.1, conv.cri=0.000001, iter=100, verbose=TRUE)