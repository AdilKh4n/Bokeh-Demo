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


#dict = {'values':[10,20,30,40], 'names':['TP\nFP\nTN',"FN"]}
#source = dict

#p = Bar(df, 'names', values='values', title="test chart")
p = Figure(tools="", toolbar_location=None)
#p.vbar(x='values', top='names',width=0.5, bottom=0,source=source)
source = ColumnDataSource(data=dict(bins=[0, 1, 2], counts=[0,0,0]))
p.vbar(x = 'bins', bottom= 1, width=0.5, top='counts' , source=source)


kernels = ["False","True"]
select = Select(title="Select the dual value", value="False", options=kernels)

input = TextInput(title="Percentage of testing set", value="0.25")


#def update_color(attrname, old, new):
#    r.glyph.fill_color = select.value
#select.on_change('value', update_color)

def update_points(attrname, old, new):
    N = float(input.value)
    D = bool(select.value)
#    source.data = get_data(N)
    df = pd.read_csv('PredictFailure_v1.0.csv',names=['failure', 'attribute1',  'attribute2\nattribute3\nattribute4\nattribute5\nattribute6\nattribute7\nattribute8\nattribute9'],low_memory=False)
    df=df[1:]
    df['failure']=[1 if b==1 or b=='1' else 0 for b in df.failure]
    y=df.failure
    x=df.drop('failure',axis=1)
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=N)
    clf = svm.LinearSVC(loss='l2', dual=D)
    #clf = svm.SVC(kernel='linear', gamma=2)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    print("True Negative:",tn)
    print("False Positive:",fp)
    print("False Negative:",fn)
    print("True Positive:",tp)
    #newdict = {'values':[tn,fp,tn,fn], 'names':['TP\nFP\nTN',"FN"]2
    source.data =dict(bins=[0, 1, 2], counts=[tp, fp, fn])
    
input.on_change('value', update_points)
select.on_change('value', update_points)

layout = column(row(input,select, width=400), row(p))

#show(layout)

curdoc().add_root(layout)
