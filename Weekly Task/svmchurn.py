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

#loading dataavg_dist	avg_rating_by_driver	avg_rating_of_driver	avg_surge	surge_pct	trips_in_first_30_days	luxury_car_user	weekday_pct	city_Astapor	city_King's Landing	city_Winterfell	phone_Android	phone_iPhone	phone_no_phone	churn
#df = pd.read_csv('churn.csv',names=['churn', 'dataavg_dist', 'avg_rating_by_driver','avg_rating_of_driverattribute3','avg_surge','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','city_Astapor',"city_King's Landing",'city_Winterfell','phone_Android','phone_no_phone'],low_memory=False)
df = pd.read_csv('churn.csv')

columns = ['luxury_car_user','avg_dist','city_Astapor',"city_KingsLanding",'phone_Android','phone_iPhone']
df1 = pd.DataFrame(df, columns=columns)

# removing the first row of data as it was name of column
# there were two 0 and two 1, so grouped them together
#df['failure']=[1 if b==1 or b=='1' else 0 for b in df.failure]
#divided data into target and features
y = df['churn']
#x=df.drop('churn',axis=1)

#split the data into training and testing sets
x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)
#For standardizing data


print("             @@@@@@@@@@@Original Accuracy and Results@@@@@@@@@@@\n")
#clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,min_samples_leaf=5)
#clf = svm.SVC(kernel='linear', gamma=0.7, C=1)
clf = svm.LinearSVC(random_state=0)
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
print("Accuracy =", accuracy_score(y_test_original,predictions))
print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
print("True Negative",tn)
print("False Positive",fp)
print("False Negative",fn)
print("True Positive",tp)
print("\n")
print("Recall: ", tp/(tp+fn))
print("Precision: ", tp/(tp+fp))

#print("Accuracy= ",accuracy_score(predictions,y_test_original))
#print("Classes printed by model= ",np.unique(predictions))
#print("Classification Report=\n",classification_report(y_test_original,predictions))
