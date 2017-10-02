from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure

import logging
logger = logging.getLogger('test')

def on_selection_change(attr, old, new):
    logger.info('selection change')

source = ColumnDataSource(data=dict(bins=[0, 1, 2], counts=[1, 10, 20]))
doc = curdoc()

p = figure(tools="box_select,tap")
source.on_change('selected', on_selection_change)

b = p.vbar(x = 'bins', bottom=0, width=0.5, top='counts' , source=source)
# c = p.circle(x='bins', y='counts', source=source)

doc.add_root(p)