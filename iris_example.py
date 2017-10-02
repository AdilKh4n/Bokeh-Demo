
from numpy.random import random
import pandas as pd
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import Select, TextInput
from sklearn import svm, datasets
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

p = Figure(tools="", toolbar_location=None)
#p.vbar(x='values', top='names',width=0.5, bottom=0,source=source)
source = ColumnDataSource(data=dict(bins=[0, 1, 2, 3], counts=[1, 10, 20, 30]))
p.vbar(x = 'bins', bottom= 1, width=0.5, top='counts' , source=source)


#kernels = ["linear","rbf","poly"]
#select = Select(title="Select the kernel", value="linear", options=kernels)

input = TextInput(title="Percentage of testing set", value="0.25")

#def update_color(attrname, old, new):
#    r.glyph.fill_color = select.value
#select.on_change('value', update_color)

def update_points(attrname, old, new):
    N = float(input.value)

    iris = datasets.load_iris()

    x = iris.data[:, :2]
    y = iris.target

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=N)
    C = 1.0  # SVM regularization parameter
    #clf = svm.SVC(kernel='linear', C=C)
    clf = svm.LinearSVC(loss='l2', dual=False)
    #      svm.LinearSVC(C=C),
    #      svm.SVC(kernel='rbf', gamma=0.7, C=C),
    #      svm.SVC(kernel='poly', degree=3, C=C))
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    print("True Negative:",tn)
    print("False Positive:",fp)
    print("False Negative:",fn)
    print("True Positive:",tp)
    #newdict = {'values':[tn,fp,tn,fn], 'names':['TP','FP','TN',"FN"]2
    source.data =dict(bins=[0, 1, 2, 3], counts=[tp, fp, fn, tn])
    # p.yaxis.bounds = (0,100000)

input.on_change('value', update_points)
#select.on_change('value', update_points)

layout = column(row( input, width=400), row(p))



curdoc().add_root(layout)
