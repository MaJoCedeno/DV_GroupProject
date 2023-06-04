import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Dataset Processing
url = 'https://github.com/MaJoCedeno/DVGroup/raw/main/'

df_access1 = pd.read_excel( url + 'dataset_payments.xlsx', sheet_name='Access1', engine='openpyxl')


# Building our Graphs
labels_access1 = ['Very easy', 'Fairly easy', 'Fairly difficult', 'Very difficult', "Don't know"]
values_access1 = df_access1.values.flatten()

data_access1 = dict(type='pie', labels=labels_access1, values=values_access1)

layout_access1 = go.Layout(title='Ease of access to cash withdrawals in the euro area in 2022',
    annotations=[
        dict(
            x=1.15,
            y=1,
            font=dict(
                size=50
 )) ])

fig_access1 = go.Figure(data=[data_access1], layout=layout_access1)

#Access 2 - Graphs
df_access2 = pd.read_excel( url + 'dataset_payments.xlsx', sheet_name='Access2', engine='openpyxl')
replace_columns_access2 = {
    'Difficult (SPACE I)': 'Fairly Difficult',
    'Difficult (SPACE II)': 'Very Difficult'
}
df_access2 = df_access2.rename(columns=replace_columns_access2)

data_access2 = []

difficult_type = df_access2.columns
difficult_type = ['Fairly Difficult','Very Difficult']

for Difficult in difficult_type:
    data_access2.append(dict(type='bar',
                     x=df_access2['COUNTRY'],
                     y=df_access2[Difficult],
                     name=Difficult,
                     showlegend=True
                  )
               )
layout_access2 = dict(title=dict(
                        text='Share of respondents perceiving access to cash withdrawals as fairly difficult or very difficult, by country'
                        ),
                  xaxis=dict(title='Fairly Difficult vs Very Difficult'),
                  yaxis=dict(title='Percentage of perceiving access by country')
                  )
fig_access2= go.Figure(data=data_access2, layout=layout_access2)
# The App itself

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1('My First DashBoard'),

    html.Div(children='Example of Access 1'),

    dcc.Graph(
        id='example-graph',
        figure=fig_access1
    ),
    dcc.Graph(
        id='example-graph',
        figure=fig_access2
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)