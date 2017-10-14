from os.path import dirname, join

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import sqlite3 as sql

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, MultiSelect, TextInput
from bokeh.io import curdoc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
#from bokeh.sampledata.movies_data import movie_path

desc = Div(text=open(join(dirname(__file__), "index.html")).read(), width=800)

features = MultiSelect(title="Features",
               options=open(join(dirname(__file__), 'features.txt')).read().split())

fruits = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
counts = [5, 3, 4, 2]

source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

p = figure(x_range=fruits, plot_height=350, toolbar_location=None, title="Counts")
p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
       line_color='white',fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

df = pd.read_csv('/Users/adilkhan/Documents/CS Fall 16/CS297/Bokeh-Demo/EmbedWebsite/churn.csv')

div = Div(text="""Your <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The
remaining div arguments are <b>width</b> and <b>height</b>. For this example, those values
are <i>200</i> and <i>100</i> respectively.""",
width=200, height=100)

''' 
# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], title=[], year=[], revenue=[], alpha=[]))

hover = HoverTool(tooltips=[
    ("Title", "@title"),
    ("Year", "@year"),
    ("$", "@revenue")
])

p = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, tools=[hover])
p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")


def select_movies():
    genre_val = genre.value
    director_val = director.value.strip()
    cast_val = cast.value.strip()
    selected = movies[
        (movies.Reviews >= reviews.value) &
        (movies.BoxOffice >= (boxoffice.value * 1e6)) &
        (movies.Year >= min_year.value) &
        (movies.Year <= max_year.value) &
        (movies.Oscars >= oscars.value)
    ]
    if (genre_val != "All"):
        selected = selected[selected.Genre.str.contains(genre_val)==True]
    if (director_val != ""):
        selected = selected[selected.Director.str.contains(director_val)==True]
    if (cast_val != ""):
        selected = selected[selected.Cast.str.contains(cast_val)==True]
    return selected

'''
def update():
    fval = features.value   
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

controls = [features]
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
curdoc().title = "Movies"