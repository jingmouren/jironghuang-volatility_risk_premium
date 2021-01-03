#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:39:24 2020

@author: jirong
"""

import pandas as pd
import numpy as np
import time
import random
import os
import datetime as dt
import yaml
import re
import quandl
import yfinance as yf
import matplotlib.pyplot as plt
import util as ut
import datetime
from bootstrapindex import bootstrapindex

class vrp_research(object):
        	  		 			     			  	   		   	  			  	
    def __init__(self, vix_cap_range, snp_cap_range,\
                 num_samples_per_period, min_sample_size, prop_block_bootstrap,\
                 days_block, starting_index
                 ):  

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
        
        self.data=None

        self.vix_cap_range=vix_cap_range  
        self.snp_cap_range=snp_cap_range  
        self.index=None
        self.num_samples_per_period=num_samples_per_period 
        self.min_sample_size=min_sample_size
        self.prop_block_bootstrap=prop_block_bootstrap 
        self.days_block=days_block 
        self.starting_index=starting_index        
              
        pass
    
    def get_data(self):
        
        """
        Obtain csv data and from yfinance
        """             
        
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
        vrp_data = pd.read_csv('vrp' + ".csv", parse_dates=['Date'], date_parser=dateparse)
        vrp_data.index = vrp_data['Date']
                      
        #Read yfinance
        price_series = ut.get_adj_open_close(tickers = ['SVXY', 'VXX', '^VIX', '^GSPC'], start_date = '2020-08-03', end_date = '2020-12-29', api = 'yfinance')                
        price_series = price_series[['SVXY_adj_close_price', 'VXX_adj_close_price', '^GSPC_adj_close_price','^VIX_adj_close_price']]
        
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
        vrp_hist = pd.read_csv('vrp_historical_data' + ".csv", parse_dates=['Date'], date_parser=dateparse)
        vrp_hist.index = vrp_hist['Date']
        vrp_hist.drop(['Date'], axis=1, inplace = True)
        vrp_hist = vrp_hist[['SVXY_adj_close_price', 'VXX_adj_close_price', '^GSPC_adj_close_price','^VIX_adj_close_price']]
        
        vrp = vrp_hist.append([price_series])     
        vrp.columns = ['svxy', 'vxx', 'gspc', 'vix'] 
        
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
        vix3m = pd.read_csv('VIX3M.csv', parse_dates = ['DATE'], date_parser = dateparse)
        vix3m.index = vix3m.DATE
        vix3m = vix3m[['CLOSE']]
        vix3m.columns = ['vix3m']    
        
        vrp = vrp.join(vix3m)    
        
        #Generate returns streams
        vrp['svxy_ret'] = 1 * vrp['svxy'].pct_change(1)          
        vrp['vxx_ret'] = 0.5 * vrp['vxx'].pct_change(1)           

        self.data = vrp           
        
        pass
    
      
    #Generate based on a range
    def generate_vix_signal(self):          
        
        """
        Generating VIX term structure signal
        """          
        
        self.data['signal_strength'] = 1 - (self.data['vix'] / self.data['vix3m']) #0-25%. Scale to 100%
       
        for i in self.vix_cap_range:
            
            j = i/100
            
            self.data['signal_strength_adj_' + str(i)] = self.data['signal_strength']/j        
            self.data['signal_strength_adj_forward_' + str(i)] = self.data['signal_strength_adj_' + str(i)].shift(periods=1)
         
            #Cap at 100%
            self.data['signal_strength_adj_forward_' + str(i)] = np.where((self.data['signal_strength_adj_forward_' + str(i)] > 1), 1, self.data['signal_strength_adj_forward_' + str(i)])
            self.data['signal_strength_adj_forward_' + str(i)] = np.where((self.data['signal_strength_adj_forward_' + str(i)] < -1), -1, self.data['signal_strength_adj_forward_' + str(i)])
            
            #Returns streams
            self.data['termstruct_returns_' + str(i)] = 0
            self.data['termstruct_returns_' + str(i)] = np.where((self.data['signal_strength_adj_forward_' + str(i)] < 0), (self.data['vxx_ret'] * self.data['signal_strength_adj_forward_' + str(i)] * (-1)), self.data['termstruct_returns_' + str(i)])
            self.data['termstruct_returns_' + str(i)] = np.where((self.data['signal_strength_adj_forward_' + str(i)] > 0), (self.data['svxy_ret'] * self.data['signal_strength_adj_forward_' + str(i)]), self.data['termstruct_returns_' + str(i)])
            
        pass

    #Generate based on a range    
    def generate_snp_signal(self):
        
        """
        Generating signal based on lagged VIX value and S&P volatility
        """                  
        
        self.data['GSPC_SD'] = self.data['gspc'].pct_change(1).rolling(21).std() * (252 ** 0.5) * 100
        self.data['vix_forward'] = self.data['vix'].shift(periods=21)
        self.data['vrp'] = self.data['vix_forward'] - self.data['GSPC_SD']
        
        for cap in self.snp_cap_range:
            
            vrp_cap_name = 'vrp_cap'+ str(cap)
            self.data[vrp_cap_name] = self.data['vrp']            
            self.data[vrp_cap_name] = np.where((self.data[vrp_cap_name] > cap) & (self.data[vrp_cap_name] > 0), cap, self.data[vrp_cap_name])
            self.data[vrp_cap_name] = np.where((self.data[vrp_cap_name] < -cap) & (self.data[vrp_cap_name] < 0), -cap, self.data[vrp_cap_name])
            
            #Downshift
            self.data['vrp_cap_forward'+ str(cap)] = self.data[vrp_cap_name].shift(periods=1)            
            self.data['prop_capital'+ str(cap)] = self.data['vrp_cap_forward'+ str(cap)]/cap
            
            #Returns streams
            self.data['snpvol_returns_' + str(cap)] = 0
            self.data['snpvol_returns_' + str(cap)] = np.where((self.data['prop_capital'+ str(cap)] < 0), (self.data['vxx_ret'] * self.data['prop_capital'+ str(cap)] * (-1) ), self.data['snpvol_returns_' + str(cap)])
            self.data['snpvol_returns_' + str(cap)] = np.where((self.data['prop_capital'+ str(cap)] > 0), (self.data['svxy_ret'] * self.data['prop_capital'+ str(cap)]), self.data['snpvol_returns_' + str(cap)])
                           
        pass    
  

    def generate_boostrap_periods(self):       

        """
        Generating walk forward block bootstrap indexes
        """   
        
        bootstrap = bootstrapindex(self.data, window='expanding', 
                                    num_samples_per_period=self.num_samples_per_period, 
                                    min_sample_size=self.min_sample_size, 
                                    prop_block_bootstrap=self.prop_block_bootstrap, 
                                    days_block=self.days_block, 
                                    starting_index = self.starting_index
                                    )        
        
        bootstrap.create_dictionary_window_n_bootstrap_index()
        self.index = bootstrap.expanding_windows_w_bootstrap_info
        
        pass
    
    def extract_period(self,period,bootstrap_index):
        
        """
        Extract single period
        """         
        
        start_index = self.index[period]['bootstrap_index']['start_index'][bootstrap_index]
        end_index = self.index[period]['bootstrap_index']['end_index'][bootstrap_index]
        
        data_sub = self.data.iloc[start_index:end_index,]
              
        return data_sub

    
    def compute_perf_single_period(self, period):
        
        """
        Compute performance in single period (in-sample)
        """          
               
        #Initialize dictionary-->change column names to be flexible
        sharpe_term_col_names = ['sharpe_term'+str(i) for i in self.vix_cap_range]
        sortino_term_col_names = ['sortino_term'+str(i) for i in self.vix_cap_range]
        drawdown_term_col_names = ['drawdown_term'+str(i) for i in self.vix_cap_range]
        returns_term_col_names = ['returns_term'+str(i) for i in self.vix_cap_range]
        
        sharpe_snp_col_names = ['sharpe_snp'+str(i) for i in self.snp_cap_range]
        sortino_snp_col_names = ['sortino_snp'+str(i) for i in self.snp_cap_range]
        drawdown_snp_col_names = ['drawdown_snp'+str(i) for i in self.snp_cap_range]
        returns_snp_col_names = ['returns_snp'+str(i) for i in self.snp_cap_range]
        
        column_names = ["period","bootstrap_index"] + sharpe_term_col_names + sortino_term_col_names + drawdown_term_col_names + returns_term_col_names + sharpe_snp_col_names + sortino_snp_col_names + drawdown_snp_col_names + returns_snp_col_names
                
        def compute_stats_single_period_single_index(index):
        
            res = pd.DataFrame(columns = column_names)
            
            data_sub = self.extract_period(period, index)
            
            res_dict = {}
            
            res_dict['period'] = period
            res_dict['bootstrap_index'] = index
            
            #If term. Loop through range of term parameters
            for i in self.vix_cap_range:
                res_dict['sharpe_term'+str(i)] = ut.get_sharpe(data_sub['termstruct_returns_' + str(i)])
                res_dict['sortino_term'+str(i)] = ut.get_sortino(data_sub['termstruct_returns_' + str(i)])                
                res_dict['drawdown_term'+str(i)] = ut.get_max_drawdown(data_sub['termstruct_returns_' + str(i)])                
                res_dict['returns_term'+str(i)] = ut.get_compound_returns(data_sub['termstruct_returns_' + str(i)])  
                res_dict['returns_drawdown_term'+str(i)] = res_dict['returns_term'+str(i)]/res_dict['drawdown_term'+str(i)]
            
            #If snp. Loop through range of snp parameters
            for i in self.snp_cap_range:
                res_dict['sharpe_snp'+str(i)] = ut.get_sharpe(data_sub['snpvol_returns_' + str(i)])
                res_dict['sortino_snp'+str(i)] = ut.get_sortino(data_sub['snpvol_returns_' + str(i)])                
                res_dict['drawdown_snp'+str(i)] = ut.get_max_drawdown(data_sub['snpvol_returns_' + str(i)])                
                res_dict['returns_snp'+str(i)] = ut.get_compound_returns(data_sub['snpvol_returns_' + str(i)])                
                res_dict['returns_drawdown_snp'+str(i)] = res_dict['returns_snp'+str(i)]/res_dict['drawdown_snp'+str(i)]
                            
            #Append to data-frame
            res = res.append(res_dict,ignore_index=True) 
            
            return res
        
        perf_bootstrap = [compute_stats_single_period_single_index(i) for i in range(self.num_samples_per_period)]
        
        perf_bootstrap = pd.concat(perf_bootstrap, axis = 0)
        
        return perf_bootstrap

    #With optimization framework    
    def compute_perf_mult_rule_single_period(self):
        
        """
        Compute performance of across all periods (in-sample)
        """                  
        
        perf = [self.compute_perf_single_period(i) for i in range(1, (1+len(self.index)) )]
        perf_bootstrap = pd.concat(perf, axis = 0)
        
        return perf_bootstrap
    
        
    #In-sample and out-of-sample
    def walk_forward_compilation(self, term_snp = 'term',param_list=[10,10,10,10,10,10,20,15]):
        
        """
        Compute performance of across all periods (out-of-sample)
        
        :param term_snp: VIX term structure or Lagged VIX/Snp Vol signals
        :param param_list: Optimized parameter generated in each in-sample period        
        """         
       
        perf_stats = pd.DataFrame(columns = ['period','param','sharpe','sortino'])
        combined_data = []
        
        for period in range(1, 1+len(self.index)):
        
            start_index = self.index[period]['out_sample_index'][0] - 1 #Some bug in indexing
            end_index = self.index[period]['out_sample_index'][1]

            if term_snp == 'term':        
                data_sub = self.data.iloc[start_index:end_index,]['termstruct_returns_'+str(param_list[(period-1)])]
            elif term_snp == 'snp':
                data_sub = self.data.iloc[start_index:end_index,]['snpvol_returns_'+str(param_list[(period-1)])]
            
            data_sub = data_sub.to_frame()

            data_sub['term_snp'] = term_snp
            data_sub['param'] = param_list[(period-1)]
            data_sub['period'] = period
            data_sub.columns = ['returns','term_snp','param','period']
            
            #Combined returns of different parameter
            combined_data.append(data_sub)
            
            #Append overall perf stats
            perf_dict = {}            
            perf_dict['period'] = period
            perf_dict['param'] = param_list[(period-1)]
            perf_dict['sharpe'] = ut.get_sharpe(data_sub['returns'])
            perf_dict['sortino'] = ut.get_sortino(data_sub['returns'])
            perf_dict['drawdown'] = ut.get_max_drawdown(data_sub['returns'])
            perf_dict['returns'] = ut.get_compound_returns(data_sub['returns'])
            perf_dict['returns_drawdown'] = perf_dict['returns']/perf_dict['drawdown']                   
            perf_stats = perf_stats.append(perf_dict,ignore_index=True)            
            
        walk_forward_returns = pd.concat(combined_data, axis = 0)
                
        return walk_forward_returns, perf_stats
          
    
if __name__ == "__main__":                     
    pass




