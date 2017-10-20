from os.path import dirname, join

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as sql

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet
from bokeh.models.widgets import Slider, MultiSelect, TextInput
from bokeh.io import curdoc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
#from sklearn import ensemble
#from bokeh.sampledata.movies_data import movie_path

#desc = Div(text=open(join(dirname(__file__), "index.html")).read(), width=800)
desc = Div(text="""<h1>An Interactive Explorer for Churn Data</h1>

<p>
Select the features to be included in the SVM Model
</p>
<p>
Prepared by <b>Adil Khan</b>.
</p>
Github Link: <a>https://github.com/AdilKh4n/Bokeh-Demo/tree/master/EmbedWebsite</a>

<br/>""")

features = MultiSelect(title="Features",
               options=open(join(dirname(__file__), 'features.txt')).read().split())
text_input = TextInput(value="print('hello')", title="Your Code:",height = 100)


#open(join(dirname(__file__), 'code.txt')).read()
df = pd.read_csv('/Users/adilkhan/Documents/CS Fall 16/CS297/Bokeh-Demo/EmbedWebsite/churn.csv')

div = Div(text="""Your <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The
remaining div arguments are <b>width</b> and <b>height</b>. For this example, those values
are <i>200</i> and <i>100</i> respectively.""",
width=200, height=100)

columns =['avg_dist', 'avg_rating_by_driver','avg_rating_of_driver','avg_surge','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','city_Astapor',"city_KingsLanding",'city_Winterfell','phone_Android','phone_no_phone']
#columns = ['luxury_car_user','avg_dist','city_Astapor',"city_KingsLanding",'phone_Android','phone_iPhone']
df1 = pd.DataFrame(df, columns=columns)
y = df['churn']
X_new = SelectKBest(chi2, k=5).fit_transform(df1, y)

x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(X_new,y,test_size=0.25)
#For standardizing data

clf = svm.LinearSVC(random_state=0)
#clf = RandomForestClassifier()
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
#print("Accuracy =", accuracy_score(y_test_original,predictions))
#print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()


fruits = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
#fruits = [tp, fp, tn, fn]
counts = [tp, fp, tn, fn]

source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

p = figure(x_range=fruits, plot_height=350, toolbar_location=None, title="Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
       line_color='white',fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

labels = LabelSet(x='fruits', y='counts', text='counts', level='glyph',
        x_offset=-15, y_offset=0, source=source, render_mode='canvas')
p.add_layout(labels)      
       
p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)


def update():
    fval = features.value
    print(fval)   
    text = text_input.value
    print(text)
    df1 = pd.DataFrame(df, columns=fval)
    y = df['churn']
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(df1,y,test_size=0.25)
    clf = svm.LinearSVC(random_state=0)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("Accuracy =", accuracy_score(y_test_original,predictions))
    #print(np.unique(predictions))
    tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
    source.data =dict(fruits=fruits, counts=[tp, fp, tn, fn])
    p.title.text = "Model Accuracy %f" % accuracy_score(y_test_original,predictions)

controls = [features,text_input]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [inputs,div,p]
])

#update()  # initial load of the data
 
curdoc().add_root(l)
curdoc().title = "Churn"