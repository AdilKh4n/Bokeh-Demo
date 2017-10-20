from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import TextInput

output_file("text_input.html")

text_input = TextInput(value="Hello", title="Label:")

show(widgetbox(text_input))