import numpy as np
import pandas as pd
import yfinance as yf

class Optimizer:

    def __init__(self,tickers,start="2020-01-01",end="2025-01-01",rf=0.08,short=True):
        self.tickers=tickers
        self.start=start
        self.end=end
        self.rf=rf
        self.short=short
        

        df=pd.DataFrame(yf.download(tickers, start,end)).dropna()
        self.close=df.Close
        daily_ret=self.close.pct_change()

        self.ret=daily_ret.mean()*252
        self.cov=daily_ret.cov()*np.sqrt(252)




        self.optimal_weights=self.optimal()
        
        
        tmp=(self.close*self.optimal_weights).sum(axis=1)
        self.optimal_share_price=tmp/tmp.iloc[0]
        self.optimal_share_price.name="Optimized Portfolio"

        self.optimal_ret=self.optimal_share_price.pct_change().mean()*252
        self.optimal_vol=self.optimal_share_price.pct_change().std()*np.sqrt(252)
        self.optimal_sharpe=(self.optimal_ret-rf)/self.optimal_vol

        


            






    
    def optimal(self,tol=1e-4):
        rf=self.rf
        short=self.short
        
        weights=np.ones_like(self.ret)

        gradient=None
        while gradient is None or np.linalg.norm(gradient)>tol:

            var=weights.T@self.cov@weights
            sw=self.cov@weights
            sharpe=(weights.T@self.ret-rf)/np.sqrt(var)
            gradient=(self.ret*np.sqrt(var)-sharpe*sw)/var
            hessian=(((sw@self.ret.T)/np.sqrt(var)-sw@gradient.T-sharpe*self.cov)*var-(self.ret*var-sharpe*sw)@(2*weights.T@self.cov))/(var)**2
            step=np.linalg.solve(hessian,-gradient)
            weights+=step
        if short:
            return weights
        else:
            return np.maximum(weights,0)


        



        
        


    
