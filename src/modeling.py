from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import time
import seaborn as sns # data visualization library  
from statsmodels.tsa.stattools import adfuller
import plotly.figure_factory as ff
import pickle

from Data_engineering import *



X_data=pd.read_csv("src/prepare_data/X_data.csv", sep = ',')
y_data=pd.read_csv("src/prepare_data/y_data.csv", sep = ',')


X_data=X_data.set_index("Date")
y_data=y_data.set_index("Date")


# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(X_data.iloc[:,:],y_data, test_size=0.3,shuffle=False)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier()      
clr_rf = clf_rf.fit(x_train,y_train)


# Save to file in the current working directory
pkl_filename = "./src/models/RandomForestClassifier_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clr_rf, file)





