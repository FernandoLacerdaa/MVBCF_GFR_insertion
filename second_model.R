# see .txt file

library(Rcpp)
library(RcppEigen)
library(RcppDist)
library(RcppArmadillo)
library(mvtnorm)
sourceCpp("SecondModel.cpp")


#Define Helper Functions
in_cred<-function(samples, value, interval)
{
  upper_quantile<-1-(1-interval)/2
  lower_quantile<-0+(1-interval)/2

  q1<-quantile(samples, lower_quantile)
  q2<-quantile(samples, upper_quantile)

  in_cred<-ifelse(value>=q1 & value<=q2, T, F)
}

cred_width<-function(samples, interval)
{
  upper_quantile<-1-(1-interval)/2
  lower_quantile<-0+(1-interval)/2

  q1<-quantile(samples, lower_quantile)
  q2<-quantile(samples, upper_quantile)

  return(q2-q1)
}


seed_val<-as.numeric(Sys.time())
set.seed(seed_val)

#Train Data
n<-500

X1<-runif(n)
X2<-runif(n)
X3<-runif(n)
X4<-runif(n)
X5<-runif(n)
X6<-rbinom(n, 1, 0.5)
X7<-rbinom(n, 1, 0.5)
X8<-rbinom(n, 1, 0.5)
X9<-sample(c(0, 1, 2, 3, 4), n, replace=T)
X10<-sample(c(0, 1, 2, 3, 4), n, replace=T)

X<-cbind(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

Mu1<-(11*sin(pi*X1*X2)+18*(X3-0.5)^2+10*X4+12*X6+X9)*10+300
Mu2<-(9*sin(pi*X1*X2)+22*(X3-0.5)^2+14*X4+8*X6+X9)*10+300

Tau1<-(2*X4+2*X5)*10
Tau2<-(1*X4+3*X5)*10

true_propensity<-X4

Z<-rbinom(n, 1, true_propensity)

Y<-cbind(Mu1+Z*Tau1, Mu2+Z*Tau2) + mvtnorm::rmvnorm(n, c(0, 0), matrix(c(50^2, 0, 0, 50^2), nrow=2, byrow=T))

#Test Data
n_test<-1000

X1_test<-runif(n_test)
X2_test<-runif(n_test)
X3_test<-runif(n_test)
X4_test<-runif(n_test)
X5_test<-runif(n_test)
X6_test<-rbinom(n_test, 1, 0.5)
X7_test<-rbinom(n_test, 1, 0.5)
X8_test<-rbinom(n_test, 1, 0.5)
X9_test<-sample(c(0, 1, 2, 3, 4), n_test, replace=T)
X10_test<-sample(c(0, 1, 2, 3, 4), n_test, replace=T)

X_test<-cbind(X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test, X8_test, X9_test, X10_test)

Mu1_test<-(11*sin(pi*X1_test*X2_test)+18*(X3_test-0.5)^2+10*X4_test+12*X6_test+X9_test)*10+300
Mu2_test<-(9*sin(pi*X1_test*X2_test)+22*(X3_test-0.5)^2+14*X4_test+8*X6_test+X9_test)*10+300

Tau1_test<-(2*X4_test+2*X5_test)*10
Tau2_test<-(1*X4_test+3*X5_test)*10

true_propensity_test<-X4_test

Z_test<-rbinom(n_test, 1, true_propensity_test)

Y_test<-cbind(Mu1_test+Z_test*Tau1_test, Mu2_test+Z_test*Tau2_test) + mvtnorm::rmvnorm(n_test, c(0, 0), matrix(c(50^2, 0, 0, 50^2), nrow=2, byrow=T))


#estimate of propensity score
p_mod<-bart(x.train = X, y.train = Z, x.test = X_test, k=3, verbose = FALSE)
p<-colMeans(pnorm(p_mod$yhat.train))
p_test<-colMeans(pnorm(p_mod$yhat.test))

#adding to matrix
X2<-X
X2_test<-X_test
X<-cbind(X, p)
X_test<-cbind(X_test, p_test)
Z2<-cbind(Z,Z)

#set some parameters
n_tree_mu<-50
n_tree_tau<-20
n_iter<-100
n_burn<-50
num_gfr<-50

mu_val<-1
tau_val<-0.375
v_val<-1
wish_val<-1
min_val<-1

mvbcf_mod <- fast_bart(X,
                       Y,
                       Z2,
                       X2,
                       X_test, #here is the test data for the mu part of the model
                       X2_test, #here is the test data for the tau part of the model
                       0.95,
                       2,
                       0.25,
                       3,
                       diag((mu_val)^2/n_tree_mu, 2),
                       diag((tau_val)^2/n_tree_tau, 2),
                       v_val,
                       diag(wish_val, 2),
                       n_iter,
                       n_tree_mu,
                       n_tree_tau,
                       min_val,
                       num_gfr)


mvbcf_tau_preds1<-rowMeans(mvbcf_mod$predictions_tau_test[,1,-c(1:n_burn)])
mvbcf_ate1<-mean(mvbcf_tau_preds1)
mvbcf_tau_preds2<-rowMeans(mvbcf_mod$predictions_tau_test[,2,-c(1:n_burn)])
mvbcf_ate2<-mean(mvbcf_tau_preds2)

mvbcf_pehe1<-sqrt(mean((Tau1_test-mvbcf_tau_preds1)^2))
mvbcf_pehe2<-sqrt(mean((Tau2_test-mvbcf_tau_preds2)^2)) 

ate1 <- mean(Tau1_test)
ate2 <- mean(Tau2_test)

print(mvbcf_ate1)
print(ate1)

print(mvbcf_ate2)
print(ate2)