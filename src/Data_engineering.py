from pandas_datareader import data

from sklearn.metrics import classification_report
# Plotting 
import matplotlib.pyplot as plt
import seaborn
import matplotlib.mlab as mlab

# Statistical calculation
from scipy.stats import norm


from numpy import log as ln
import numpy as np
from math import *
import datetime
import seaborn as sn

from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std


from scipy.stats import shapiro



import pandas as pd 
print(abs(-100))

import sys
import plotly.graph_objs as go  

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Check if local computer has the library yfinance. If not, install. Then Import it.
#!{sys.executable} -m pip install yfinance # Check if the machine has yfinance, if not, download yfinance
import yfinance as yf # Import library to access Yahoo finance stock data
# Add financial information and indicators 
#!{sys.executable} -m pip install ta # Download ta
from ta import add_all_ta_features # Library that does financial technical analysis

import  ta 
import plotly.express as px




from sklearn.metrics import silhouette_samples,silhouette_score
import seaborn as sns


# use thiiiiiiiiiiiiiiiiiiiiiiiiiis
class Data_engineering:

    ticker = ""
    data=pd.DataFrame()
    data=pd.DataFrame()
    data=pd.DataFrame()

    def __init__(self,ticker):
        self.ticker=ticker

            
    def Var_STD_Mean(self):
        #df2_tidy = self.data.rename(columns = {self.ticker: 'Adj Close'}, inplace = False)
        df2_tidy = self.data.rename(columns = {self.ticker: 'sp500'}, inplace = False)
        self.stats["mean"]=df2_tidy.mean()
        self.stats["Std.Dev"]=df2_tidy.std()
        self.stats["Var"]=df2_tidy.var()
        self.variance =  self.stats["Var"]
        return self.stats

    def add_Indicator(self,tickerSymbol,start,end):
      tickerData = yf.Ticker(tickerSymbol)

      # Create historic data dataframe and fetch the data for the dates given. 
      df = tickerData.history(start = '2012-01-01', end = '2021-01-01')

      data_indic=pd.DataFrame()
      #RSI
      data_indic["RSI_14"]=ta.momentum.RSIIndicator(close= df['Close'],  window= 14, fillna = False).rsi() 
      data_indic["RSI_5"]=ta.momentum.RSIIndicator(close= df['Close'],  window= 5, fillna = False).rsi() 
      data_indic["RSI_7"]=ta.momentum.RSIIndicator(close= df['Close'],  window= 7, fillna = False).rsi() 
      data_indic["RSI_50"]=ta.momentum.RSIIndicator(close= df['Close'],  window= 50, fillna = False).rsi() 
      #ROC
      data_indic["ROC_12"]=ta.momentum.roc(close= df['Close'],  window= 12, fillna = False)
      data_indic["ROC_25"]=ta.momentum.roc(close= df['Close'],  window= 25, fillna = False)
      data_indic["ROC_200"]=ta.momentum.roc(close= df['Close'],  window= 200, fillna = False)
      #MACD
      data_indic["MACD"]=ta.trend.MACD(close= df['Close'], window_slow= 26, window_fast = 12, window_sign = 9, fillna = False).macd() 
      #TRIX
      data_indic["TRIX_15"]=ta.trend.TRIXIndicator(close= df['Close'],  window= 15, fillna = False).trix()
      data_indic["TRIX_30"]=ta.trend.TRIXIndicator(close= df['Close'],  window= 30, fillna = False).trix()
      #MA
      data_indic["MA_10"]=ta.volatility.bollinger_mavg(close= df['Close'],  window= 10, fillna = False)
      data_indic["MA_20"]=ta.volatility.bollinger_mavg(close= df['Close'],  window= 20, fillna = False)
      data_indic["MA_50"]=ta.volatility.bollinger_mavg(close= df['Close'],  window= 50, fillna = False)
      data_indic["MA_100"]=ta.volatility.bollinger_mavg(close= df['Close'],  window= 100, fillna = False)
      data_indic["MA_200"]=ta.volatility.bollinger_mavg(close= df['Close'],  window= 200, fillna = False)
      #EMA
      data_indic["EMA_12"]=ta.trend.EMAIndicator(close= df['Close'],  window= 12, fillna = False).ema_indicator()
      data_indic["EMA_26"]=ta.trend.EMAIndicator(close= df['Close'],  window= 26, fillna = False).ema_indicator()
      data_indic["EMA_50"]=ta.trend.EMAIndicator(close= df['Close'],  window= 50, fillna = False).ema_indicator()
      data_indic["EMA_200"]=ta.trend.EMAIndicator(close= df['Close'],  window= 200, fillna = False).ema_indicator()
      #ADX
      data_indic["ADX"]=ta.trend.adx(high= df['High'], low=df['Low'], close=df['Close'], window=14, fillna=False)
      #CCI
      data_indic["CCI"]=ta.trend.CCIIndicator(high= df['High'], low=df['Low'], close=df['Close'], window= 20, constant = 0.015, fillna = False).cci()
      #VI
      data_indic["NegativeVolume"]=ta.volume.NegativeVolumeIndexIndicator(close=df['Close'], volume=df['Volume'], fillna = False).negative_volume_index()

      data_indic["ADI"]=ta.volume.acc_dist_index(high= df['High'], low=df['Low'], close=df['Close'],  volume=df['Volume'], fillna=False)
      data_indic["MFI"]=ta.volume.money_flow_index(high= df['High'], low=df['Low'], close=df['Close'],  volume=df['Volume'],window=14, fillna=False)
      data_indic["OBV"]=ta.volume.on_balance_volume( close=df['Close'],  volume=df['Volume'], fillna=False)
      indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
      # Add Bollinger Bands features
      data_indic["Bollinger_mavg"] = indicator_bb.bollinger_mavg()
      data_indic["Bollinger_hband"] = indicator_bb.bollinger_hband()
      data_indic["Bollinger_lband"] = indicator_bb.bollinger_lband()

      # Add Bollinger Band high indicator
      data_indic["Bollinger_hband_indicator"] = indicator_bb.bollinger_hband_indicator()

      # Add Bollinger Band low indicator
      data_indic["Bollinger_lband_indicator"] = indicator_bb.bollinger_lband_indicator()
      return data_indic

    def return_func(self,df,day,column_name):
      ff=df.copy()
      ff["label_return_"+str(day)+"_"+column_name[9:]]=ff[column_name].pct_change(day).shift(-day)
      return ff
   
    def new_label(self,AMZN_,plafond_gain,plancher_perte,time_barrier,column_name):
      AMZN_1=AMZN_.copy()
      AMZN_1['return']=AMZN_1[column_name].pct_change()

      AMZN_1=AMZN_1.dropna()
      l=[]
      plafond_gainv=plafond_gain
      plancher_pertev=plancher_perte
      alpha=0
      for i in range( AMZN_1.shape[0]-time_barrier):
          for j in range(i,i+time_barrier):
              if AMZN_1["return"][j] > 0.02:
                  alpha=1
                  l.append(alpha)
                  break
              
              elif AMZN_1["return"][i] < -0.01:
                  alpha=-1
                  l.append(alpha)
                  break
              else :
                  if j== i+time_barrier-1:
                      k=(AMZN_1["Adj CloseAMZN"][i+time_barrier-1]-AMZN_1["Adj CloseAMZN"][i])/AMZN_1["Adj CloseAMZN"][i]
                      if k>1:
                          alpha=1
                          l.append(alpha)
                      else :
                          alpha=-1  
                          l.append(alpha)

      new_lab=pd.DataFrame(data=l, columns=["new_ label_return"],index=AMZN_1.index[:-time_barrier])
      df=pd.concat([AMZN_1,new_lab], axis=1)
      df=df.dropna()
      return df

    def prepare_data2(self,AMZN_,SP500,Indicateur,plafond_gain,plancher_perte,time_barrier):
      new_data=self.new_label(AMZN_,plafond_gain,plancher_perte,time_barrier,"Adj CloseAMZN")
      new_dataSP500=self.return_func(SP500,1,"Adj CloseSP500")
      new_dataSP500=new_dataSP500.drop(columns="Adj CloseSP500")
      new_dataSP500=new_dataSP500.drop(columns="CloseSP500")
      new_dataSP500=new_dataSP500.drop(columns="VolumeSP500")
      #new_dataSP500.columns=new_dataSP500.column+"_sp500"
    
 
      #new_data=new_data.drop(columns="Adj CloseAMZN")
      new_data=new_data.drop(columns="CloseAMZN")
      new_data = new_data.rename(columns = {'Adj CloseAMZN': 'close', }, inplace = False)

      new_data_f=pd.concat([new_dataSP500,new_data], axis=1)
                
      return pd.merge(Indicateur, new_data_f, left_index=True, right_index=True)         

    """def prepare_data(self,AMZN_,SP500,Indicateur,return_val, day_pred):

 

      new_data=self.return_func(AMZN_,day_pred,"Adj CloseAMZN")
      new_dataSP500=self.return_func(SP500,day_pred,"Adj CloseSP500")
      new_dataSP500=new_dataSP500.drop(columns="Adj CloseSP500")
      new_dataSP500=new_dataSP500.drop(columns="CloseSP500")
      new_dataSP500=new_dataSP500.drop(columns="VolumeSP500")
      #new_dataSP500.columns=new_dataSP500.column+"_sp500"
    
      
      new_data=new_data.dropna()
      for i in new_data.columns[-1:]:
        print(i)
        for j in range(new_data.shape[0]):
          if new_data[i][j]> return_val:
            new_data[i][j]=1
          else:
            new_data[i][j]=0   

      #new_data2=self.return_func(AMZN_,1,"Adj CloseAMZN")
      #new_data2=new_data2.dropna()
      #new_data["daily_return"]=new_data2.iloc[:,-1].to_list()
      new_data=new_data.drop(columns="Adj CloseAMZN")
      new_data=new_data.drop(columns="CloseAMZN")

      new_data_f=pd.concat([new_dataSP500,new_data], axis=1)
                
      return pd.merge(Indicateur, new_data_f, left_index=True, right_index=True)"""

    def one_days(self,the_data):
     # the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)

      return the_data

    """def two_days(self,the_data):
     # the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:]
      data2=the_data[1:]
      data2.index=data1.index[:-1]
      data2.columns=data2.columns+"_day_2"
      aa=pd.concat([data1,data2], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa"""
      
    def two_days(self,the_data):
    # the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-1]
      data2=the_data[1:]

    
      data1.index=data2.index
      data1.columns=data1.columns+"_day_2"
      aa=pd.concat([data1,data2], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa  


    def three_days(self,the_data):
      #the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-2]
      data2=the_data[1:-1]
      data3=the_data[2:]

      data1.index=data3.index
      data2.index=data3.index
      data1.columns=data1.columns+"_day_3"
      data2.columns=data2.columns+"_day_2"
      dd=pd.concat([data1,data2,data3], axis=1,)
   
      #dd=dd.drop(dd.columns[-2], axis=1)
      return dd
    def four_days(self,the_data):
      #the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-3]
      data2=the_data[1:-2]
      data3=the_data[2:-1]
      data4=the_data[3:]

      data2.index=data4.index
      data3.index=data4.index
      data1.index=data4.index
      data3.columns=data3.columns+"_day_2"
      data2.columns=data2.columns+"_day_3"
      data1.columns=data1.columns+"_day_4"
      aa=pd.concat([data1,data2,data3,data4], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa
      
    def five_days(self,the_data):
      #the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-4]
      data2=the_data[1:-3]
      data3=the_data[2:-2]
      data4=the_data[3:-1]
      data5=the_data[4:]

      data2.index=data5.index
      data3.index=data5.index
      data4.index=data5.index
      data1.index=data5.index
      data4.columns=data4.columns+"_day_2"
      data3.columns=data3.columns+"_day_3"
      data2.columns=data2.columns+"_day_4"
      data1.columns=data1.columns+"_day_5"
      aa=pd.concat([data1,data2,data3,data4,data5], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa
      
    def six_days(self,the_data):
     # the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-5]
      data2=the_data[1:-4]
      data3=the_data[2:-3]
      data4=the_data[3:-2]
      data5=the_data[4:-1]
      data6=the_data[5:]

      data2.index=data6.index
      data3.index=data6.index
      data4.index=data6.index
      data5.index=data6.index
      data1.index=data6.index
      data5.columns=data5.columns+"_day_2"
      data4.columns=data4.columns+"_day_3"
      data3.columns=data3.columns+"_day_4"
      data2.columns=data2.columns+"_day_5"
      data1.columns=data1.columns+"_day_6"
      aa=pd.concat([data1,data2,data3,data4,data5,data6], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa
      
    def seven_days(self,the_data):
     # the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-6]
      data2=the_data[1:-5]
      data3=the_data[2:-4]
      data4=the_data[3:-3]
      data5=the_data[4:-2]
      data6=the_data[5:-1]
      data7=the_data[6:]

      data2.index=data7.index
      data3.index=data7.index
      data4.index=data7.index
      data5.index=data7.index
      data6.index=data7.index
      data1.index=data7.index
      data6.columns=data6.columns+"_day_2"
      data5.columns=data5.columns+"_day_3"
      data4.columns=data4.columns+"_day_4"
      data3.columns=data3.columns+"_day_5"
      data2.columns=data2.columns+"_day_6"
      data1.columns=data1.columns+"_day_7"
      aa=pd.concat([data1,data2,data3,data4,data5,data6,data7], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa
      
    def eight_days(self,the_data):
      #the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-7]
      data2=the_data[1:-6]
      data3=the_data[2:-5]
      data4=the_data[3:-4]
      data5=the_data[4:-3]
      data6=the_data[5:-2]
      data7=the_data[6:-1]
      data8=the_data[7:]

      data2.index=data8.index
      data3.index=data8.index
      data4.index=data8.index
      data5.index=data8.index
      data6.index=data8.index
      data7.index=data8.index
      data1.index=data8.index
      data7.columns=data7.columns+"_day_2"
      data6.columns=data6.columns+"_day_3"
      data5.columns=data5.columns+"_day_4"
      data4.columns=data4.columns+"_day_5"
      data3.columns=data3.columns+"_day_6"
      data2.columns=data2.columns+"_day_7"
      data1.columns=data1.columns+"_day_8"
      aa=pd.concat([data1,data2,data3,data4,data5,data6,data7,data8], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa
      
    def nine_days(self,the_data):
      #the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-8]
      data2=the_data[1:-7]
      data3=the_data[2:-6]
      data4=the_data[3:-5]
      data5=the_data[4:-4]
      data6=the_data[5:-3]
      data7=the_data[6:-2]
      data8=the_data[7:-1]
      data9=the_data[8:]

      data2.index=data9.index
      data3.index=data9.index
      data4.index=data9.index
      data5.index=data9.index
      data6.index=data9.index
      data7.index=data9.index
      data8.index=data9.index
      data1.index=data9.index
      data8.columns=data8.columns+"_day_2"
      data7.columns=data7.columns+"_day_3"
      data6.columns=data6.columns+"_day_4"
      data5.columns=data5.columns+"_day_5"
      data4.columns=data4.columns+"_day_6"
      data3.columns=data3.columns+"_day_7"
      data2.columns=data2.columns+"_day_8"
      data1.columns=data1.columns+"_day_9"
      aa=pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa
    def ten_days(self,the_data):
     # the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
      the_data=the_data.dropna(axis=0)
      data1=the_data[0:-9]
      data2=the_data[1:-8]
      data3=the_data[2:-7]
      data4=the_data[3:-6]
      data5=the_data[4:-5]
      data6=the_data[5:-4]
      data7=the_data[6:-3]
      data8=the_data[7:-2]
      data9=the_data[8:-1]
      data10=the_data[9:]

      data2.index=data10.index
      data3.index=data10.index
      data4.index=data10.index
      data5.index=data10.index
      data6.index=data10.index
      data7.index=data10.index
      data8.index=data10.index
      data9.index=data10.index
      data1.index=data10.index
      data9.columns=data9.columns+"_day_2"
      data8.columns=data8.columns+"_day_3"
      data7.columns=data7.columns+"_day_4"
      data6.columns=data6.columns+"_day_5"
      data5.columns=data5.columns+"_day_6"
      data4.columns=data4.columns+"_day_7"
      data3.columns=data3.columns+"_day_8"
      data2.columns=data2.columns+"_day_9"
      data1.columns=data1.columns+"_day_10"
      aa=pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10], axis=1,)
      #aa=aa.drop(aa.columns[-2], axis=1)
      return aa

    def get_ticker_data(self,tickerSymbol,start,end,column_name):
            AMZN_ = data.get_data_yahoo(tickerSymbol,start,end,interval="d")#['Adj Close']
            #AMZN_=AMZN_.to_frame()
            #AMZN_ = AMZN_.rename(columns={'Adj Close': column_name})
            AMZN_.columns=AMZN_.columns+column_name
            return AMZN_

    def Data_eng(self,i,Indicateur,AMZN_,SP500,plafond_gain,plancher_perte,time_barrier):
            #the_data=self.prepare_data(AMZN_,SP500,Indicateur,threshold,pred)
            the_data=self.prepare_data2(AMZN_,SP500,Indicateur,plafond_gain,plancher_perte,time_barrier)
            
         
            switcher={
                    1: self.one_days(the_data),
                    2: self.two_days(the_data),
                    3: self.three_days(the_data),
                    4: self.four_days(the_data),
                    5: self.five_days(the_data),
                    6: self.six_days(the_data),
                    7: self.seven_days(the_data),
                    8: self.eight_days(the_data),
                    9: self.nine_days(the_data),
                    10:self.ten_days(the_data)
                    }
            
            return switcher.get(i) 

    """def Data_eng2(self,i,tickerSymbol,start,end,threshold,pred):
            Indicateur=self.add_Indicator(tickerSymbol,start,end)
            AMZN_ = data.get_data_yahoo(tickerSymbol,start,end,interval="d")['Adj Close']
            AMZN_=AMZN_data.to_frame()
            AMZN_ = AMZN_.rename(columns={'Adj Close': 'AMZN Adj Close'})
            if pred==1:
              return self.one_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==2:  
              return self.two_days((AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==3:
              return self.three_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==4:
              return self.four_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==5:
              return self.five_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==6:
              return self.six_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==7:
              return self.sevne(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==8:
              return self.eight_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==9:
              return self.nine_days(AMZN_,SP500,Indicateur,threshold,pred)
            elif pred==10:
              return self.ten_days(AMZN_,SP500,Indicateur,threshold,pred)"""




