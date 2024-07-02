#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pandas as pd


# load the training dataset
bike_data = pd.read_csv("data/bike_data.csv")

## Train a Regression model ##
# Separate features and labels
X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
#print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
# Fit a lasso model on the training set

#model = GradientBoostingRegressor().fit(X_train, y_train)
#z = np.polyfit(y_test, predictions, 1)
#p = np.poly1d(z)
# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

def fit_model(selected_model):
    if selected_model == 'LinearRegression':
        return LinearRegression().fit(X_train, y_train)
    elif selected_model == 'Lasso':
        return Lasso().fit(X_train, y_train)
    elif selected_model == 'GradientBoostingRegressor':
        return GradientBoostingRegressor().fit(X_train, y_train)
    elif selected_model == 'RandomForestRegressor':
        return RandomForestRegressor().fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {selected_model}")
        
        
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    dash.dependencies.Input('model-dropdown', 'value')
)
def update_scatter_plot(selected_model):
    model = fit_model(selected_model)
    predictions = model.predict(X_test)
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    r2 = r2_score(y_test, predictions)

    # Create scatter plot
    scatter_data = [
        {
            'x': y_test,
            'y': predictions,
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Predictions',
        },
        {
            'x': y_test,
            'y': p(y_test),  # Regression line (actual labels)
            'mode': 'lines',
            'type': 'scatter',
            'name': 'Regression Line',
            'line': {'color': 'blue', 'dash': 'dash'},
        },
    ]

    scatter_layout = {
        'title': 'Scatter Plot: Actual vs. Predicted Labels \n(R^2: '+ str(round(r2,3))+")",
        'xaxis': {'title': 'Actual Labels'},
        'yaxis': {'title': 'Predicted Labels'},
    }

    return {'data': scatter_data, 'layout': scatter_layout}

# Define the layout of the app
app.layout = html.Div([
    html.Label('Model'),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Linear Regression', 'value': 'LinearRegression'},
            {'label': 'Lasso', 'value': 'Lasso'},
            {'label': 'GradientBoostingRegressor', 'value': 'GradientBoostingRegressor'},
            {'label': 'RandomForestRegressor', 'value': 'RandomForestRegressor'},
        ],
        value='LinearRegression',  # Set the default value
        style={'marginRight': '10px'}
    ),
    dcc.Graph(id='scatter-plot'),
])



if __name__ == '__main__':
    app.run_server(debug=True)

