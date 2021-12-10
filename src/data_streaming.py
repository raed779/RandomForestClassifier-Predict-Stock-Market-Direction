from Data_engineering import *



amzn_data__=Data_engineering("amzn")

Indicateur=amzn_data__.add_Indicator('amzn','2012-01-01','2021-01-01')
AMZN_ = amzn_data__.get_ticker_data('amzn','2012-01-01','2021-01-01','AMZN') 
SP500 = amzn_data__.get_ticker_data('^GSPC','2012-01-01','2021-01-01',"SP500") 


Indicateur.to_csv("src/prepare_data/Indicateur.csv")
AMZN_.to_csv("src/prepare_data/AMZN_.csv")
SP500.to_csv("src/prepare_data/SP500.csv")

print("ok rag")