import pandas as pd
import numpy as np

from numpy.random import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn import svm
from bokeh.io import curdoc, show
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import Select, TextInput
from bokeh.charts import Bar

#output_file("color_change.html")


#def get_data(N):
#    return dict(x=random(size=N), y=random(size=N), r=random(size=N) * 0.03)


#dict = {'values':[10,20,30,40], 'names':['TP','FP','TN',"FN"]}
#   source = ColumnDataSource(data=dict(values=[0, 1, 2], timing=[1, 10, 20]))

data = {
    'values': ['TP', 'FN', 'TN', 'FP'],
    'timing': [10,20,30,40]
}


#p = Figure(tools="", toolbar_location=None)
#p.vbar(x='values', top='names',width=0.5, bottom=0,source=source)
#df = ColumnDataSource(data=dict(bins=[0, 1, 2], counts=[1, 10, 20]))
#p = Bar(df,label = 'bins', values='counts', title="test chart")

bar = Bar(data, values='timing', label='values', title="py", legend='top_right', width=400)

input = TextInput(title="Percentage of testing set", value="0.25")

#def update_color(attrname, old, new):
#    r.glyph.fill_color = select.value
#select.on_change('value', update_color)

def update_points(attrname, old, new):
    N = float(input.value)
#    source.data = get_data(N)
    df = pd.read_csv('PredictFailure_v1.0.csv',names=['failure', 'attribute1',  'attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute8','attribute9'])
    df=df[1:]
    df['failure']=[1 if b==1 or b=='1' else 0 for b in df.failure]
    y=df.failure
    x=df.drop('failure',axis=1)
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=N)
    clf = svm.LinearSVC(loss='l2', penalty='l1', dual=False)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    print("True Negative:",tn)
    print("False Positive:",fp)
    print("False Negative:",fn)
    print("True Positive:",tp)
    #newdict = {'values':[tn,fp,tn,fn], 'names':['TP','FP','TN',"FN"]
    newdict = {
    'values': ['TP', 'FN', 'TN', 'FP'],
    'timing': [tp,fp,fn,tn]
    }
    data.update(newdict)
    print(data)

input.on_change('value', update_points)

layout = column(row( input, width=400), row(bar))

#show(layout)

curdoc().add_root(layout)
