from bokeh.charts import Bar
from bokeh.io import save
from bokeh.plotting import Figure, ColumnDataSource
import pandas as pd

dict = {'values':[10,20,30,40], 'names':['TP','FP','TN',"FN"]}
df = pd.DataFrame(dict)

p = Bar(df, 'names', values='values', title="test chart")

dict1 = {'values':[10,20,30,40], 'names':['TP','FP','TN',"FN"]}
source = ColumnDataSource(dict1)

#p = Bar(df, 'names', values='values', title="test chart")
r = Figure(tools="", toolbar_location=None)
r.vbar(x='values', top='names',width=0.5, bottom=0,source=source)

save(r,'test3.html')
