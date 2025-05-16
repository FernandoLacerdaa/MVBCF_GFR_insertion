### Inserting a Grow-From-Root parameter into the R library **MVBCF** (Multivariate Bayesian Causal Forests)

The **BCF** (Bayesian Causal Forests) implementation at [BCF](https://github.com/StochasticTree/stochtree/blob/main/R/bcf.R) has a **num_gfr** “warm-start” parameter.  

* **num_gfr** is basically the number of warm-up iterations where trees are grown greedily (no randomness) to a reasonable depth. Thus, the parameter helps in faster convergence and a better initial fit.  
* It also results in a “smoother” MCMC due to reduced burn-in time, because the trees already capture basic patterns.  
* So **num_gfr** basically gives the model a “smarter start” before fine-tuning with Markov Chain Monte Carlo.

Unfortunately, the MVBCF lib [MVBCF](https://github.com/Nathan-McJames/MVBCF-Paper/blob/main/GitHub_DGP1.R) does not have its own **num_gfr** like a regular BCF. The goal of this repo is to track my attempts (in different ways) to insert **num_gfr** into the MVBCF lib. Every code here is still under development and will definitely get better over time.


