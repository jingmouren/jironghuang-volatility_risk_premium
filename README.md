# volatility_risk_premium
Using term structure to exploit volatility risk premium

## Disclaimer
None of the contents constitute an offer to sell, a solicitation to buy, or a recommendation or endorsement for any security or strategy, nor does it constitute an offer to provide investment advisory services Past performance is no indicator of future performance Provided for informational purposes only All investments involve risk, including loss of principal

Note: I have a vested interest in this strategy. Variant of the strategy has been deployed since start of Sep-2020

## Summary

- Out-of-sample walk forward analysis sharpe ratio is 0.89 and sortino ratio is 1.21
- Strategy exhibits a negative skew of -0.88 and fat tail kurtosis of 8.18.
- Such a strategy has the potential to perform well in volatile environment-evident by the explosive returns during 2020 (~ +100%) and normal environment (> 100% in 2017)
- The strategy exhibits a low correlation of 0.25 to SPY

## Introduction

- In this notebook, I will explore term structure of VIX i.e. VIX and VIX3M as a viable signal for volatility trading.

Data was sourced from couple of sources,

- Expired long VIX futures ETF, VXX and short VIX futures ETF, SVXY from ETF providers adjusted for reverse splits
- Current VXX and SVXY ETFs adjusted for reverse splits
- VIX and VIX3M from yahoo finance and CBOE respectively
- SPY from yahoo finance used for benchmarking

## Approach

### Signal generation

To generate a signal, I will compare term structure of VIX against VIX3M.

- If VIX < VIX3m, a long position on SVXY is initiated the next day.
- If VIX > VIX3m, a long position on VXX is initiated the next day.


### Continuous signal

To smooth out returns stream, I convert VIX/VIX3M into a continuous signal,

- self.data['signal_strength'] = 1 - (self.data['vix'] / self.data['vix3m'])
- A normalizing denominator, j (0.1, 0.15, 0.2, 0.25 parameters are tested for optimization) is further applied to convert continuous signal to -100% to 100% signal
- self.data['signal_strength_adj_' + str(i)] = self.data['signal_strength']/j
- The ensuing signal is the proportion of available capital (or fix capital) used in position sizing.

### Walk forward and Block boostrapping analysis

- To pick the best parameter, I performed a walk forward and block boostrapping analysis using a package that I developed (https://pypi.org/project/bootstrapindex/)
- Walk Forward Analysis optimizes on a training set; test on a period after the set and then rolls it all forward and repeats the process. There are multiple out-of-sample periods and the combined results can be analyzed.
- To facilitate Walk Forward Analysis, the package produces start and end of block bootstrap indexes within each training set data chunk.
- Block bootstrap indexes basically represents continuous chunks of time series indexes that are sampled with replacement within a training set data chunk.

## How to use this repository

- I developed a volatility risk premium research class for this strategy.
- You may initialize the strategy class as follow,

strategy = vrp_research(vix_cap_range = [10, 15, 20, 25],\
                        snp_cap_range = [10, 15, 20, 25],\
                        num_samples_per_period=100,\
                        min_sample_size=100,prop_block_bootstrap=0.10,\
                        days_block=252,starting_index=22  
                        ) 
                        
"""
Constructor for VRP class

:param data: data-frame holding data and signals
:param vix_cap_range: list of vix_cap for continuous signals
:param snp_cap_range: list of snp_cap for continuous signals
:param num_samples_per_period: num_samples_per_period in walk forward block bootstrapping
:param min_sample_size: minimum sample size in each block bootstrap sample in walk forward block bootstrapping
:param prop_block_bootstrap: Proportion of dataset used for block bootstrapping
:param days_block: Number of days used in each out-of-sample block. 
:param starting_index: Starting index in data-frame for whole analysis. Can be randomized to avoid butterfly effect.
:return: returns VRP class
"""           

- Pls look at the jupyter notebook (volatility_risk_premium.ipynb) to understand how to use the class.
- Pls look at the documentation on the vrp_research_class (vrp_research_class_documentation.pdf).
- Note that this is not an engineering project but more of a research project. Error handling and engineering features are not included. Should any of the repository for deployment purpose, these should be included.
           