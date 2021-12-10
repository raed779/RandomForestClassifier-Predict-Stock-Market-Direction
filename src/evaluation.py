
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



pkl_filename = "./src/models/RandomForestClassifier_model.pkl"

# Load from file
with open(pkl_filename, 'rb') as file:
    clf_rf = pickle.load(file)



ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
d=sns.heatmap(cm,annot=True,fmt="d")
d.figure.savefig("confusion_matrix.png")


report = classification_report(y_test,clf_rf.predict(x_test))
print('Model accuracy', accuracy_score(y_test,clf_rf.predict(x_test), normalize=True))
print(report)


import plotly.express as px
col_sorted_by_importance=clf_rf.feature_importances_.argsort()
feat_imp=pd.DataFrame({
    'cols':x_train.columns[col_sorted_by_importance],
    'imps':clf_rf.feature_importances_[col_sorted_by_importance]
})


fih=px.bar(feat_imp, x='cols', y='imps')
fih.write_image("feature_importances_.png", engine="kaleido")


