import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn import svm

#loading data
df = pd.read_csv('PredictFailure_v1.0.csv',names=['failure', 'attribute1', 'attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute8','attribute9'],low_memory=False)
# removing the first row of data as it was name of column
df=df[1:]
# there were two 0 and two 1, so grouped them together
df['failure']=[1 if b==1 or b=='1' else 0 for b in df.failure]
#divided data into target and features
y=df.failure
x=df.drop('failure',axis=1)
#split the data into training and testing sets
x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=0.25)
#For standardizing data


print("             @@@@@@@@@@@Original Accuracy and Results@@@@@@@@@@@\n")
#clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,min_samples_leaf=5)
clf = svm.SVC(kernel='rbf', gamma=0.7, C=1)
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
print("Accuracy =", accuracy_score(y_test_original,predictions))
print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
print("true negative",tn)
print("false positive",fp)
print("false negative",fn)
print("true positive",tp)
#print("Accuracy= ",accuracy_score(predictions,y_test_original))
#print("Classes printed by model= ",np.unique(predictions))
#print("Classification Report=\n",classification_report(y_test_original,predictions))
