import pandas as pd
import numpy as np


from bokeh.io import curdoc
from bokeh.models.widgets import Div, Select, TextInput, Slider
from bokeh.layouts import layout, widgetbox, column
from bokeh.models import Label
from bokeh.plotting import figure

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Scikit has many annoying warnings...
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

##############################################################################
###                          SET UP HEADER/FOOTER                          ###
##############################################################################

# Page header and footer
page_header = Div(text = """
            <style>
            h1 {
                margin: 1em 0 0 0;
                color: #2e484c;
                font-family: 'Julius Sans One', sans-serif;
                font-size: 1.8em;
                text-transform: uppercase;
            }
            a:link {
                font-weight: bold;
                text-decoration: none;
                color: #0d8ba1;
            }
            a:visited {
                font-weight: bold;
                text-decoration: none;
                color: #1a5952;
            }
            a:hover, a:focus, a:active {
                text-decoration: underline;
                color: #9685BA;
            }
            p {
                text-align: justify;
                text-justify: inter-word;
                /*font: "Libre Baskerville", sans-serif;
                width: 90%;
                max-width: 940;*/
            }
            small {
                color: #424242;
            }

            p.big {
                margin-bottom: 0.8cm;
            }

            </style>
            <h1>Pile Capacity Predictor (BETA)</h1>
            <p>
            This online tool features a <em>Support Vector Regressor</em> to predict the axial load capacity of pile foundations given soil type, average SPT-N values, pile type and open/closed end condition, pile cross sectional area, circumference and length. The process is outlined in:
            <br><br>
            <blockquote>
            Machairas, N. P., and Iskander, M. G. (2018). “An Investigation of Pile Design Utilizing Advanced Data Analytics.” <em>Proceedings of the International Foundations Congress and Equipment Expo 2018</em>, ADSC-The International Association of Foundation Drilling, DFI (Deep Foundations Institute), G-I (Geo-Institute of American Society of Civil Engineers), and PDCA (Pile Driving Contractors Association), March 5-10, 2018, Orlando, Florida.
            </blockquote>
            </p>
            <br />
            <h3>DISCLAIMER</h3>
            <p>
            This tool is offered without any warranties about the accuracy of the predicted capacity. The predicted capacity is a result of approximation by scientific methodologies. The authors' sole intent is to further advance the field of Geotechnical Engineering and are not offering this online tool as a design aid. <strong>Use to learn and experiment, do not design piles based on the numbers you get below.</strong>
            </p>
            <br />
            <hr />
            <br />
            """
,width = 940)

page_footer = Div(text = """
            <br>
            <hr>
            <br>
            <div>
                <div align="left">
                  <a href="http://engineering.nyu.edu/academics/departments/civil" target="_blank">
                    <img src="media/tandon_cue_color.png" style="float:left" onerror="this.src='http://drive.google.com/uc?export=view&id=0B2vCK8uO_I7kWFYwbHM3UXVNalk'" width="25%">
                  </a>
                </div>
                <div align="right" style="color:gray">
                    Developed by:<br>
                    <a href="https://github.com/nickmachairas" target="_blank">Nick Machairas</a><br />
                    <a href="http://engineering.nyu.edu/people/magued-g-iskander/" target="_blank">Magued Iskander</a>
                </div>
            </div>
            """
,width = 940)




##############################################################################
###                             SET UP INPUTS                              ###
##############################################################################


# Soil Type dropdown
soil_type = Select(title = "Select Predominant Soil Type:",
                   value = "Sand",
                   options = ['Sand','Clay','Mixed']
                   ,width=180)

# Average SPT-N entry
# avg_N = TextInput(title="Enter Average SPT-N count:",
#                   value=str(50),width=180)
avg_N = Slider(start=7, end=120,
               value=50, step=1,
               title="Select Average SPT-N count",
               width=180)

# Create list of soil widgets
soil_controls = [Div(text="<strong>SOIL PROPERTIES</strong>"),
                 soil_type,
                 Div(text=""),
                 avg_N]

# Create soil_inputs widgetbox
soil_inputs = widgetbox(*soil_controls, width=250)


# Pile Type dropdown
pile_type = Select(title = "Select Pile Type:",
                   value = "Steel",
                   options = ['Steel','Concrete','Composite'])

# Pile Open end dropdown
open_end = Select(title = "Open Ended?",
                   value = "No",
                   options = ['Yes','No'])

# Cross sectional area
# cross_area = TextInput(title="Enter cross sectional area (in2):",
#                   value=str(16.1))
cross_area = Slider(start=5, end=1200,
                    value=16, step=1,
                    title="Select cross sectional area (in2)")

# Circumference
# circ = TextInput(title="Enter circumference (in):",
#                   value=str(44.0))
circ = Slider(start=30, end=120,
                    value=44, step=1,
                    title="Select circumference (in)")

# Length
# length = TextInput(title="Enter length (ft):",
#                   value=str(22.5))
length = Slider(start=9, end=173,
                    value=22.5, step=1,
                    title="Select length (ft)")

# Create list of pile widgets
pile_controls1 = [Div(text="<strong>PILE PROPERTIES</strong>"),
                 pile_type,
                 open_end]

# Create pile_inputs widgetbox
pile_inputs1 = widgetbox(*pile_controls1, width=200)


# Create list of pile widgets
pile_controls2 = [Div(text="<p class='big'> " " </p>"),
                  cross_area,
                  Div(text=""),
                  circ]

# Create pile_inputs widgetbox
pile_inputs2 = widgetbox(*pile_controls2, width=220)


# Create list of pile widgets
pile_controls3 = [Div(text="<p class='big'> " " </p>"),
                  length]

# Create pile_inputs widgetbox
pile_inputs3 = widgetbox(*pile_controls3, width=225)



##############################################################################
###                           RUN CALCULATIONS                             ###
##############################################################################

## SVM Regression

# Load data
df = pd.read_csv('ifcee2018_predict.csv')

# Data preprocessing:
X_svm = df.ix[:, df.columns != 'davisson']
y_svm = df.davisson.values

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_svm = X_scaler.fit_transform(X_svm)
y_svm = y_scaler.fit_transform(y_svm)


# Create the SVR object w/ appropriate coefficients
svr = SVR(kernel='rbf',
          C=16.2377,
          epsilon=0.01,
          gamma=0.3393)

# Split the data
X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(
                            X_svm, y_svm, test_size=0.30, random_state=737)

# Train the model
svr.fit(X_svm_train, y_svm_train)

# Check MSE and R2
# svr_MSE = mean_squared_error(y_scaler.inverse_transform(svr.predict(X_svm_test)),
#                              y_scaler.inverse_transform(y_svm_test))
# print("SVR MSE: {:.5f}".format(svr_MSE))
# svr_train_R2 = svr.score(X_svm_train, y_svm_train)
# svr_test_R2 = svr.score(X_svm_test, y_svm_test)
# print('SVR train-R2: {:.5f}'.format(svr_train_R2))
# print('SVR test-R2: {:.5f}'.format(svr_test_R2))

prediction_scaled = svr.predict(X_svm_train[0])
prediction = y_scaler.inverse_transform(prediction_scaled)[0]
# print(X_svm_train[0])
# print(df.columns)
# print('*_*_*_*_*_*:',prediction)


##############################################################################
###                              SET UP PLOT                               ###
##############################################################################



plot = figure(x_range=(0, 10),
              y_range=(0, 10),
              plot_height=120,
              plot_width=940)
plot.axis.visible = False
plot.toolbar.logo = None
plot.toolbar_location = None
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None

label = Label(x=5, y=5,
              text='{:.2f} kips'.format(prediction),
              text_font_size='70pt',
              text_color='#7f7f7f',
              text_align='center',
              text_baseline='middle')
plot.add_layout(label)




##############################################################################
###                            SET UP CALLBACK                             ###
##############################################################################



# Define a callback function: callback
def callback(attr, old, new):

    # Read the current values
    new_soil_type = soil_type.value
    new_avg_N = avg_N.value
    new_pile_type = pile_type.value
    new_open_end = open_end.value
    new_cross_area = cross_area.value
    new_circ = circ.value
    new_length = length.value

    # create dummies for soil_type
    if new_soil_type == 'Sand':
        new_soil_type = [0,0,1]
    elif new_soil_type == 'Clay':
        new_soil_type = [1,0,0]
    else:
        new_soil_type = [0,1,0]

    # Create dummies for pile_type
    if new_pile_type == 'Steel':
        new_pile_type = [0,0,1]
    elif new_pile_type == 'Concrete':
        new_pile_type = [0,1,0]
    else:
        new_pile_type = [1,0,0]

    # Get dummy for open_end
    if new_open_end == 'Yes':
        new_open_end = [1]
    else:
        new_open_end = [0]


    # Compile new inputs
    new_inputs = (new_soil_type + new_pile_type + [float(new_avg_N)] +
                  new_open_end + [float(new_cross_area)] + [float(new_circ)] +
                  [float(new_length)])
    new_inputs = np.array(new_inputs)

    # Scale new inputs
    new_inputs_scaled = X_scaler.transform(new_inputs)

    new_prediction_scaled = svr.predict(new_inputs_scaled)
    new_prediction = y_scaler.inverse_transform(new_prediction_scaled)[0]


    if new_prediction > 0:
        label.text = '{:.2f} kips'.format(new_prediction)
    else:
        label.text = 'N/A'

# Call the callback function on update of these fields
for i in [soil_type, avg_N, pile_type,
          open_end,cross_area,circ,length]:
    i.on_change('value', callback)



##############################################################################
###                             SET UP LAYOUT                              ###
##############################################################################

# Set up initial page layout
page_layout = layout([[page_header],
                      [soil_inputs,pile_inputs1,pile_inputs2,pile_inputs3],
                      [Div(text="<br><h2>RESULT</h2>")],
                      [plot],
                      [page_footer]],
                      width = 940)


# Add the page layout to the current document
curdoc().add_root(page_layout)
curdoc().title = "Capacity Predictor"

# run with:
# bokeh serve --show ifcee2018_predict.py

# run forever on server with:
# nohup bokeh serve ifcee2018_predict.py --allow-websocket-origin cue3.engineering.nyu.edu:5012 --host cue3.engineering.nyu.edu:5012 --port 5012
