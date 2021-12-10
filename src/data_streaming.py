from Data_engineering import *



amzn_data__=Data_engineering("amzn")

Indicateur=amzn_data__.add_Indicator('amzn','2012-01-01','2021-01-01')
AMZN_ = amzn_data__.get_ticker_data('amzn','2012-01-01','2021-01-01','AMZN') 
SP500 = amzn_data__.get_ticker_data('^GSPC','2012-01-01','2021-01-01',"SP500") 


Indicateur.to_csv("src/prepare_data/Indicateur.csv")
AMZN_.to_csv("src/prepare_data/AMZN_.csv")
SP500.to_csv("src/prepare_data/SP500.csv")




def getWeights(d,lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1,1) 
    return w
def plotWeights(dRange, lags, numberPlots):
    weights=pd.DataFrame(np.zeros((lags, numberPlots)))
    interval=np.linspace(dRange[0],dRange[1],numberPlots)
    for i, diff_order in enumerate(interval):
        weights[i]=getWeights(diff_order,lags)
    weights.columns = [round(x,2) for x in interval]
    fig=weights.plot(figsize=(15,6))
    plt.legend(title='Order of differencing')
    plt.title('Lag coefficients for various orders of differencing')
    plt.xlabel('lag coefficients')
    #plt.grid(False)
    plt.show()
def ts_differencing(series, order, lag_cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients
    
    weights=getWeights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:] 


def Differencing2(test1):
  liss=[]
  for i in test1.columns:
    if "MA" in i or "EMA" in i or "ROC" in i or "NegativeVolume" in i or "ADI" in i or "Bollinger_mavg" in i or "Bollinger_lband" in i or "OBV" in i or "Bollinger_hband" in i:
      #print(i)
      liss.append(i)
  f1=liss

  liss2=[]
  for i in test.columns:
    if "High" in i or "Low" in i or "Open" in i or "VolumeA" in i :
      #print(i)
      liss2.append(i)


  for i in  test1.columns:
    if  i.startswith("EMA") :
      test1[i][20:]=ts_differencing(test1[i],0.9,20).values
    if i.startswith("MA"):
      test1[i][20:]=ts_differencing(test1[i],0.98,20).values
    
  for s in  f1:    
    test1[s][20:]=ts_differencing(test1[s],0.9,20).values
    test1=test1.dropna()
  for s1 in  liss2:    
    #test1[liss2]=test1[liss2].diff(periods=1)
    test1[s1][20:]=ts_differencing(test1[s1],0.9,20).values
    test1=test1.dropna()
  return test1 [50:]

features=['RSI_14', 'RSI_5', 'RSI_7', 'RSI_50', 'ROC_12', 'ROC_25', 'ROC_200',
       'MACD', 'TRIX_15', 'TRIX_30', 'MA_10', 'MA_20', 'MA_50', 'MA_100',
       'MA_200', 'EMA_12', 'EMA_26', 'EMA_50', 'EMA_200', 'ADX', 'CCI',
       'NegativeVolume', 'ADI', 'MFI', 'OBV', 'Bollinger_mavg',
       'Bollinger_hband', 'Bollinger_lband', 'Bollinger_hband_indicator',
       'Bollinger_lband_indicator', "return1"]


amzn_data__=Data_engineering("amzn")
test=amzn_data__.Data_eng(2,Indicateur,AMZN_,SP500,0.02,-0.01,10)
test=test.dropna()
test=test.drop(columns="close")
df=Differencing2(test)
#df=df.iloc[:200,]
features=df.columns.to_list()
y_data=df[features[-1]]
liss3=[]
for i in df.columns:
  if "return" in i  :
    #print(i)
    liss3.append(i)

for i in liss3[:]:

  features.remove(i)    

X_data=df[features]


print(X_data.head(5))



X_data.to_csv("src/prepare_data/X_data.csv")
y_data.to_csv("src/prepare_data/y_data.csv")

