### Inserting Grow From Roots parameter into the R library MVBCF (Multi Variable Bayesian Causal Forests)

# The BCF (Bayesian Causal Forests) at [BCF](https://github.com/StochasticTree/stochtree/blob/main/R/bcf.R) has a num_gfr "warm-start" parameter 

# num_gfr is basically a the number of warm-up iterations where trees are grown greedily (no randomness) to a reasonable depth. Thus, the parameters helps in a faster convergence and better initial fit. 

# It also results in a "smoother" MCMC due to reduced "burn-in" time because the trees already capture basic patterns.

# So the num_gfr basically gives the model a "smarter start" before fine-tuning with Markov Chain Monte Carlo.

# Unfortunetly the MVBCF lib [MVBCF](https://github.com/Nathan-McJames/MVBCF-Paper/blob/main/GitHub_DGP1.R) does not have its own way of a num_gfr like a regular BCF, so the goal of this repo is to track my attempts of trying (from different ways) to insert the num_gfr into the MVBCF lib. Every code here still is under development and will definetly get better over time.

