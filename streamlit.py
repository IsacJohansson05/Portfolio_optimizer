import streamlit as st
import plotly.express as px
import plotly.io as pio
import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
from backend import Optimizer
pio.templates.default="seaborn"

st.sidebar.header("Portfolio Optimizer 1.0")

sp500_tickers=list(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"])

stocks = st.sidebar.multiselect("Stocks (only S&P 500):",sp500_tickers)




today=dt.date.today()


start=st.sidebar.date_input("Start Date:",today.replace(year=today.year-10))
end=st.sidebar.date_input("End Date:",today)
rf=st.sidebar.slider('Risk-free rate:', 0.0, 20.0, 5.0, format="%.1f%%") / 100
short=st.sidebar.checkbox("Allow short",False)


spy=yf.download("SPY",start,end).Close
spy=spy/spy.iloc[0]
spy_ret=float(spy.pct_change().mean()*252)
spy_vol=float(spy.pct_change().std()*np.sqrt(252))
spy_sharpe=(spy_ret-rf)/spy_vol






try:
    portfolio=Optimizer(stocks,start,end,rf,short)
    df = pd.concat([portfolio.optimal_share_price,spy], axis=1)


    fig = px.line(df,title="Optimized portfolio vs S&P 500")


    st.plotly_chart(fig)


    col1, col2, col3 = st.columns(3) 

    with col1:
        st.metric("Return:", f"{portfolio.optimal_ret*100:.2f}%", delta=f"{(portfolio.optimal_ret-spy_ret)*100:.2f}% (units)")

    with col2:
        st.metric("Volatility:", f"{portfolio.optimal_vol*100:.2f}%", delta=f"{(portfolio.optimal_vol-spy_vol)*100:.2f}% (units)")

    with col3:
        st.metric("Sharpe Ratio:", f"{portfolio.optimal_sharpe:.2f}",delta=f"{(portfolio.optimal_sharpe-spy_sharpe):.2f}")


    
    if short:
        portfolio.optimal_weights/=sum(np.maximum(portfolio.optimal_weights,0))
    else:
        portfolio.optimal_weights/=sum(portfolio.optimal_weights)


    c1,c2=st.columns([1,2])
    if not short:
        with c2:
            fig=px.pie(names=stocks,values=portfolio.optimal_weights)
            st.plotly_chart(fig)
    with c1:
        st.subheader("Optimal weights")
        weights=pd.DataFrame(portfolio.optimal_weights,index=stocks,columns=["weights"])

        weights*=100
        weights = weights.applymap("{:.2f}%".format)
        st.table(weights)

    
except ValueError:
    st.write("Choose stocks")
except IndexError:
    st.write("Input a valid time interval")