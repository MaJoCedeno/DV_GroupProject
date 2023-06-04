import numpy as np
import pandas as pd

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import math

from datetime import datetime, date

# import sklearn
# from sklearn.preprocessing import LabelEncoder

# ######################################################################################################################
# ##### ELSH #####

# ### DATASET UPLOADING ###
# Topics 1&2: Payment Preference Evolution'19vs22 & Reasons'22
path = 'https://github.com/MaJoCedeno/DVGroup/raw/main/'

# ## Dataframe Evolution
df2 = pd.read_excel(path + 'DV_DB_Evolution.xlsx', sheet_name='2', engine='openpyxl')
df5 = pd.read_excel(path + 'DV_DB_Evolution.xlsx', sheet_name='5', engine='openpyxl')
df20 = pd.read_excel(path + 'DV_DB_Evolution.xlsx', sheet_name='20', engine='openpyxl')
df22 = pd.read_excel(path + 'DV_DB_Evolution.xlsx', sheet_name='22_23', engine='openpyxl')
# df23 = pd.read_excel(path + 'DV_DB_Evolution.xlsx', sheet_name='23', engine='openpyxl')

# ## Dataframe - Access
# Pie chart
df_access1 = pd.read_excel(path + 'dataset_payments.xlsx', sheet_name='Access1', engine='openpyxl')
#  Heatmap
df_access2_1 = pd.read_excel(path + 'dataset_payments.xlsx', sheet_name='Access2_1', engine='openpyxl')
df_access2_1.set_index('COUNTRY', inplace=True)

# ### DATAFRAMES PREP ###

# ##Define a dictionary to map the country codes to country names
country_mapping = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'CY': 'Cyprus',
    'DE': 'Germany',
    'EuroArea': 'EuroArea',
    'EE': 'Estonia',
    'ES': 'Spain',
    'FI': 'Finland',
    'FR': 'France',
    'GR': 'Greece',
    'IE': 'Ireland',
    'IT': 'Italy',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'PT': 'Portugal',
    'SI': 'Slovenia',
    'SK': 'Slovakia'}


# ## DF2 preparation ###
# ##df2_copy sorted by Cash, added two columns Labels19 & Labels22
df2_copy = df2.copy().sort_values('y_2022', ascending=False).reset_index(drop=True)
df2_copy.loc[:, 'labels19'] = ['<b>' + i + '<br>' + str(round(j * 100)) + '%' for i, j in
                               zip(df2_copy['Payment_Instrument'], df2_copy['y_2019'])]
df2_copy.loc[:, 'labels22'] = ['<b>' + i + '<br>' + str(round(j * 100)) + '%' for i, j in
                               zip(df2_copy['Payment_Instrument'], df2_copy['y_2022'])]

# ##split df2_copy  into  df2_n_srt (Numbers of transactions) & df2_v_srt (values of payments)
df2_n_srt = df2_copy[df2_copy["Type_Number_or_Value"] ==
                     "Number of payments"].sort_values('y_2022', ascending=False).reset_index(drop=True)
df2_v_srt = df2_copy[df2_copy["Type_Number_or_Value"] ==
                     "Value of payments"].sort_values('y_2022', ascending=False).reset_index(drop=True)

# ## DF5 preparation ###
# ###### DataSets preparation based on df5 ###
# ##Exclude the 'EuroArea' row  and sort
df5_sorted = df5[df5['Country'] != 'EuroArea'].sort_values('Cash').reset_index(drop=True)
# ##Merge 'Diff_ 19-20' from df20 to df5
df5_sorted = df5_sorted.merge(df20[['Country', 'Diff_ 19-20']], on='Country', how='left')
# ##Map the country codes to country names
df5_sorted['CTRY'] = df5_sorted['Country'].map(country_mapping)

# ## Create two dfs: df5_n with Number of payments, df5_v w Value of payments
df5_n = df5_sorted[df5_sorted["Type_Number_or_Value"] == "Number of transactions"].reset_index(
    drop=True)  # has only Number of payments
df5_v = df5_sorted[df5_sorted["Type_Number_or_Value"] == "Value of payments"].reset_index(
    drop=True)  # has only Values of payments

# ## NUMBERS|
# Expand df5_n: X and Y coordinates for circled plot
df5_n_copy = df5_n.copy()

# ## X and Y in column + newLabels Column
df5_n_copy.loc[:, 'X'] = 1
list_y = list(range(0, len(df5_n_copy)))
list_y.reverse()
df5_n_copy.loc[:, 'Y'] = list_y
df5_n_copy.loc[:, 'labels'] = ['<b>' + i + '<br>' + str(round(j * 100)) + '%' for i, j in
                               zip(df5_n_copy['CTRY'], df5_n_copy['Cash'])]

# ##create X and Y coordinates in a circle
df5_n_copy['Degree'] = df5_n_copy.index * (360 / len(df5_n_copy))
df5_n_copy.sort_values(by='Cash', ascending=False, inplace=True)
df5_n_copy['Degree'] += 90  # Adjust the starting position of the largest circle

df5_n_copy['X_coor'] = np.cos(df5_n_copy['Degree'] * np.pi / 180)
df5_n_copy['Y_coor'] = np.sin(df5_n_copy['Degree'] * np.pi / 180)

# ## VALUES|
# Expand df5_v: X and Y coordinates for circled plot
df5_v_copy = df5_v.copy()

df5_v_copy.loc[:, 'X'] = 1
list_y2 = list(range(0, len(df5_v_copy)))
list_y2.reverse()
df5_v_copy.loc[:, 'Y'] = list_y2
df5_v_copy.loc[:, 'labels'] = ['<b>' + i + '<br>' + str(round(j * 100)) + '%' for i, j in zip(df5_v_copy['CTRY'],
                                                                                              df5_v_copy['Cash'])]
# ##create X and Y coordinates in a circle
df5_v_copy['Degree'] = df5_v_copy.index * (360 / len(df5_v_copy))
df5_v_copy.sort_values(by='Cash', ascending=False, inplace=True)
df5_v_copy['Degree'] += 90  # Adjust the starting position of the largest circle

df5_v_copy['X_coor'] = np.cos(df5_v_copy['Degree'] * np.pi / 180)
df5_v_copy['Y_coor'] = np.sin(df5_v_copy['Degree'] * np.pi / 180)

# ## Numbers & Values| Create df5_both
# concatenate  df5_v_copy and df5_n_copy into df5_both
df5_both = pd.concat([df5_v_copy, df5_n_copy]).reset_index(drop=True)
df5_both["Type_Number_or_Value"] = df5_both["Type_Number_or_Value"].replace("Number of transactions",
                                                                            "Number of payments")

###################################

# ## DF20 preparation ###
# Exclude the 'EuroArea' row for sorting
df20_sorted = df20[df20['Country'] != 'EuroArea'].sort_values('Diff_ 19-20')
# Create a new DataFrame with the 'EuroArea' row
# euro_area_row = df20[df20['Country'] == 'EuroArea']
# Append the 'EuroArea' row to the beginning of the sorted DataFrame
# df20_sorted = pd.concat([euro_area_row, df20_sorted]).reset_index(drop=True)
# Map the country codes to country names
df20_sorted['CTRY'] = df20_sorted['Country'].map(country_mapping)

# df 22 23, reasons to prefer  Cash or Card. Top-3 reasons

sorted_df = df22.sort_values(by=['Method', 'y_2022'], ascending=False)

grouped_df = sorted_df.groupby('Method')

# Create an empty DataFrame to store the top 5 values for each category
top5_dataframes = []

# Iterate over each group and select the top 3 values
for group, data in grouped_df:
    top5_data = data.head(5)  # Select the top 3 values for each group
    top5_dataframes.append(top5_data)  # Append the selected data to the list

# Concatenate the selected data frames into a single DataFrame
top5_df = pd.concat(top5_dataframes)

# Reset the index of the new DataFrame
top5_df.reset_index(drop=True, inplace=True)


# #Colors
def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal


pal_vi = get_color('viridis_r', len('df5n'))
pal_plas = get_color('plasma_r', len('df5n'))
pal_ECB = ['#003299', '#001A66', '#003D99', '#0066CC', '#4DA4FF', '#B3D9FF', '#93C6FF', '#6FA8FF', '#F58025', '#996700']
# clr_range = ['#bee9e8', '#003299', '#62b6cb', '#1b4965', '#cae9ff', '#5fa8d3']
color_range = ['#bee9e8', '#62b6cb', '#1b4965', '#cae9ff', '#5fa8d3']
pal_ecb2 = ['#1b98e0', '#003299', '#E8F1F2', '#0A81D1', '#00ffff']

# ######################################################################################################################

# #######  CREATING PLOTS #######

# ## TOPIC 1_  ############ START###
# ## Evolution of preferences for cash, 2019 vs 2022


# ##### Plot FIG20 creation ###
# ### Idiom: bar plot

# using plotly.express
fig20 = px.bar(
    df20_sorted,
    y='CTRY',
    x='Diff_ 19-20',
    orientation='h',
    text='Country',  # write country name inside a bar
    color=np.where(df20_sorted['Diff_ 19-20'] < 0, 'Negative change', 'Positive change'),
    color_discrete_map={'Negative change': '#003299', 'Positive change': '#00ffff'},  # #996700
    hover_name="CTRY",
    labels={'y_2019': '2019 RATE', 'y_2022': '2022 RATE', 'Change in pp': 'CHANGE 2019-2022 in pp',
            },
    hover_data={
        'y_2019': ':.2f',
        'y_2022': ':.2f',
        'Change in pp': True, 'CTRY': False, 'Country': False, 'Diff_ 19-20': False, }
)
# Add annotations to the bars
for i, row in df20_sorted.iterrows():
    fig20.add_annotation(
        x=row['Diff_ 19-20'],
        y=row['CTRY'],
        text=f'{row["Diff_ 19-20"] * 100:.0f}%',
        showarrow=False,
        font=dict(size=10),
        align='center',
        xshift=15 if row['Diff_ 19-20'] >= 0 else -18
    )
# Ensure Country label stays inside  a bar regardless window size
fig20.update_traces(width=0.7, textposition='inside', insidetextanchor='start')

# Set the labels and title
fig20.update_layout(
    yaxis_title='COUNTRY',
    xaxis_title='2019-2022 CHANGE',
    # title='Preference for Cash: Evolution from Year 2019 to 2022, per Country,',
    plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
    xaxis=dict(showticklabels=True, showgrid=False, gridcolor="white",
               tickformat='.0%', tickfont=dict(size=8, color="grey")),
    yaxis=dict(
        title='',
        showgrid=False, gridcolor="#fafafa",
        # title_standoff=15,
        tickson="boundaries",
        title_font=dict(family="Tableau Bold"),
        tickfont=dict(size=12, family="Tableau Book", color="black")
    ),
    barmode='relative',
    bargap=0.1,
    bargroupgap=0.1,
    showlegend=False,
    uniformtext_minsize=8,  # uniformtext_mode='hide',
    font=dict(
        family="Tableau Book",
        size=12,
        # color="RebeccaPurple"
    ),
    width=600,
    height=500
)
# fig20.show()
# ## TOPIC 1_ GRAPH 1.2 (Chart 20) ############ END ###


# ## FIGURE 'ACCESS'  | CREATION ###
# ## Idiom: Pie Chart
# ## (by Majo)
labels_access1 = ['Very easy', 'Fairly easy', 'Fairly difficult', 'Very difficult', "Don't know"]
values_access1 = df_access1.values.flatten()

data_access1 = dict(
    type='pie',
    labels=labels_access1,
    values=values_access1,
    hole=0.8,  # This property creates the donut hole with a size of 0.4 (40%)
    insidetextorientation='radial',
    textposition='auto',
    textinfo='label+percent',
)

layout_access1 = go.Layout(  # title='Ease of access to cash withdrawals in the euro area in 2022',
    annotations=[
        dict(
            x=1.15,
            y=1,
            font=dict(size=50, family='Tableau Book'),

        )], width=650, height=380,
    margin=dict(t=20, l=50, r=0, b=0),
    legend=dict(
        x=0.5,
        y=0.5,
        bgcolor='rgba(255, 255, 255, 0.8)',  # Set the background color of the legend
        yanchor='middle',
        xanchor='center',
        font=dict(family='Tableau Book')  # Set the font family for the legend
    )

)
colors = ['#bee9e8', '#62b6cb', '#1b4965', '#cae9ff', '#5fa8d3']

fig_access1 = go.Figure(data=[data_access1], layout=layout_access1)
fig_access1.update_traces(marker=dict(colors=colors))
# pie chart ACCESS _ END of creation ##########################

# ######################################################################################################################
# ############# DEFINE APP  ###
# ############# (always start with these two lines when creating DASH)###
app = dash.Dash(__name__)
server = app.server

# ############ DEFINE APP LAYOUT ###
app.layout = html.Div([
    # # #

    html.Div([

        html.Img(src='https://www.ecb.europa.eu/shared/img/logo/logo_only.svg',
                 style={'height': '8%', 'position': 'relative', 'opacity': '80%'}),
        html.Label('CONSUMER PAYMENT TRENDS | EURO AREA | 2019-2022'),

    ], id='1_div', style={'display': 'flex', 'height': '8%'}),

    # # #
    html.Div([
        html.Div(children=[

            html.Div(children=[
                html.H2('PAYMENT BEHAVIOUR & PREFERENCE'),
                html.Br(),

                html.Div([
                    dcc.Dropdown(
                        id="dropdown", options=["Number of payments", "Value of payments"],
                        value="Number of payments", clearable=False, ),

                    html.Br(),

                    html.Div([

                        html.Div([

                            html.Div([
                                html.Div([
                                    html.H2('PAYMENT METHOD | USAGE RATE | 2019 VS 2022',
                                            style={'margin-bottom': '10px'}),
                                    html.H4('Share of payment instruments used at POS', style={'margin-bottom': '0px'}),
                                    dcc.Graph(id="chart2", style={'text-align': 'center'}),
                                ], className='box'),
                                html.Br(),

                                html.Div([
                                    html.H2('CASH | PREFERENCE RATE CHANGE'),
                                    html.H4('Preference for Cash: Evolution from Year 2019 to 2022, per Country',
                                            style={'margin-bottom': '10px'}),
                                    html.H4('Is there a change in Cash preference rate?',
                                            style={'margin-bottom': '0px'}),
                                    dcc.Graph(id='bar_graph_20_2', figure=fig20)], className='box',
                                    style={'text-align': 'center',
                                           'display': 'inline-block'}),

                            ], style={'text-align': 'center'},
                            ),

                            html.Div([
                                html.H2('CASH | USAGE RATE 2022'),
                                html.H4('Share of Cash as a Payment Instrument Used at POS in 2022, per country'),
                                html.H4('Who has the highest/lowest share of cash usage?'),
                                html.Br(),
                                html.Br(),
                                html.Div([
                                    html.P(
                                        'ORDER: Circles ordered clockwise from the country with highest cash share '
                                        'to the lowest \n'
                                        'COLOR: Positive or Negative change in cash preference compared to 2019 \n '
                                        'SIZE: Circle size denotes the cash share compared to other countries',

                                        style={'font-size': '12px', 'white-space': 'pre-wrap'}),
                                ], style={'height': '8%'}),

                                html.Br(),
                                html.Br(),
                                html.Br(),
                                html.Br(),

                                dcc.Graph(id='chart5'),

                            ], style={'text-align': 'center', 'height': '20%'}
                            ),

                        ], style={'text-align': 'center', 'display': 'flex'}, className='box'
                        ),

                    ], style={'display': 'flex', }),

                    html.Div([
                        html.Br(),
                        html.Br(),
                        html.Br(),
                    ])

                ], style={'display': 'block'})

            ], style={'display': 'inline-block', }, id='Block w Share of payments and Cash used',
            ),

            html.Div([
                html.Div(children=[

                    html.Div([
                        html.H2('RATIONALE'),
                        html.H4('Reasons to prefer Cash or Card, 2022'),
                        dcc.RadioItems(
                            id='method_radio',
                            options=[{'label': 'Why Cash?', 'value': 'Cash'},
                                     {'label': 'Why Card?', 'value': 'Card'}],
                            value='Cash'),
                        dcc.Graph(id="treemap_reason"),
                    ], style={'text-align': 'center', 'display': 'inline-block'}, className='box'),

                    html.Div([
                        html.H2('CASH | ACCESS RATE'),
                        html.H4('Ease of access to cash withdrawals, 2022',
                                style={'margin-bottom': '10px'}),
                        html.H4('How easy do they find it to get to an ATM or a bank to withdraw cash?',
                                style={'margin-bottom': '2px'}),
                        dcc.Graph(id='pie_access', figure=fig_access1)
                    ], style={'text-align': 'center', 'display': 'inline-block'})
                ], style={'display': 'flex'}),

                html.Div([
                    dcc.Graph(id='heatmap_plot')
                ]),
            ], className='box')

        ], style={'display': 'inline-block'}, className='box')

    ], style={'text-align': 'center',
              'display': 'flex', 'font-size': 12,},
        id='Global block with all save for Title'
    ),
],
    style={'text-align': 'center', 'display': 'inline-block', 'font-size': 12,
           # 'background-color': 'black', 'color': 'white' # for a dark mode
           }
)


# ####################
# ## CALLBACK FIG2 ###
@app.callback(Output("chart2", "figure"), Input("dropdown", "value"))
def update_fig2(type_num_val):
    df2_dd = df2_copy.copy(deep=True)  # always create a copy of df
    mask = df2_dd["Type_Number_or_Value"] == type_num_val

    # fig2 : Two subplots showing RATE USED per Instrument: stacked barplot 2019  & stacked barplot 2022
    fig2 = make_subplots(rows=2, cols=1, row_titles=('2019', '2022'), vertical_spacing=0.0001)

    # Add the first bar plot for 2019
    fig2.add_trace(
        go.Bar(x=df2_dd[mask]['y_2019'], y=df2_dd[mask]['Type_Number_or_Value'], orientation='h',
               text=df2_dd[mask]['labels19'], textposition='inside', showlegend=False,
               marker=dict(color=df2_n_srt['Payment_Instrument'].map(
                   {k: pal_ecb2[i] for i, k in enumerate(['Cards', 'Cash', 'Other', 'Mobile app'])}))),
        row=1, col=1
    )

    # Add the second bar plot for 2022
    fig2.add_trace(
        go.Bar(x=df2_dd[mask]['y_2022'], y=df2_dd[mask]['Type_Number_or_Value'], orientation='h',
               text=df2_dd[mask]['labels22'], textposition='inside', showlegend=False,
               marker=dict(color=df2_n_srt['Payment_Instrument'].map(
                   {k: pal_ecb2[i] for i, k in enumerate(['Cards', 'Cash', 'Other', 'Mobile app'])}))),
        row=2, col=1
    )
    # Update layout and axis settings
    fig2.update_layout(width=600, height=150,
                       plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                       margin=dict(t=30, l=20, r=5, b=2),
                       title=dict(
                           text=f'{type_num_val}',
                           x=0.01,
                           y=0.90,
                           font=dict(family='Tableau Book', color='grey')
                       ),
                       showlegend=False, font_family='Tableau Book')
    fig2.update_traces(insidetextanchor='middle')

    fig2.update_xaxes(visible=False)
    fig2.update_yaxes(visible=False)

    # Update subplot titles position
    fig2.update_annotations(dict(xref='paper', x=0.001), font_color='white', font_family='Tableau Book', textangle=-90)

    return fig2


# ## CALLBACK FIG2 ### END ###
##############################

# # CALLBACK FIG 5 ###
# ## Circle with  circles _ Cash  USED Rate, 2022
@app.callback(Output("chart5", "figure"), Input("dropdown", "value"))
def update_fig5(type_num_val):
    df5_both_dd = df5_both.copy()
    mask5 = df5_both_dd["Type_Number_or_Value"] == type_num_val
    fig5 = px.scatter(df5_both_dd[mask5], x='X_coor', y='Y_coor',
                      color=df5_both_dd[mask5]['Diff_ 19-20'].apply(lambda x: 'Increase in cash preference' if x > 0
                      else 'Decrease in cash preference'),
                      color_discrete_map={'Decrease in cash preference': '#003299',
                                          'Increase in cash preference': '#00ffff'},
                      size='Cash', text='labels', size_max=40, hover_name="CTRY",
                      labels={'Cash': 'Cash Used 2022 RATE', 'Card': 'Card Used 2022 RATE',
                              'Diff_ 19-20': 'Cash Preference CHANGE 2019-2022'},
                      hover_data={

                          'Cash': ':.2f', 'Card': ':.2f',
                          'X_coor': False, 'Y_coor': False, 'CTRY': False, 'Country': False, 'labels': False,
                          'Diff_ 19-20': ':.2f'})

    fig5.update_traces(textposition='bottom center', textfont_size=10)
    fig5.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig5.update_yaxes(showgrid=False, zeroline=False, visible=False)

    fig5.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        'legend': {'x': 0.5, 'y': 0.5, 'bgcolor': 'rgba(0, 0, 0, 0)', 'yanchor': 'middle',
                                   'xanchor': 'center'}}, legend=dict(title='Color code', ),
                       width=650, height=650,
                       margin=dict(t=10, l=0, r=0, b=20),
                       title=dict(
                           text=f'{type_num_val}',
                           x=0.01,  # Place the title in the center of the figure
                           y=0.96,
                           font=dict(family='Tableau Book', color='grey')
                       ),
                       showlegend=True, uniformtext_minsize=8, font=dict(
            family="Tableau Book",
            # size= 10,
            # color="RebeccaPurple"
        ))
    return fig5


# ## FIG 22_23 TREE-MAP
@app.callback(
    Output("treemap_reason", "figure"),
    Input("method_radio", "value"))
def update_treemap(reason):
    top5_df_dd = top5_df.copy(deep=True)
    mask22 = top5_df_dd["Method"] == reason

    fig22 = px.treemap(top5_df_dd[mask22], path=[px.Constant(reason), 'Reason_modified'],
                       values='y_2022_rebalanced',
                       color='y_2022_rebalanced',
                       color_continuous_scale=color_range,
                       color_continuous_midpoint=np.average(top5_df_dd['y_2022_rebalanced'])
                       )

    # Increase the font size of the treemap labels for all levels
    fig22.update_traces(textfont=dict(size=28), selector=dict(type='treemap'), texttemplate='%{label}<br>%{value:.0%}', hovertemplate='<b>Reason</b>: %{label}')

    fig22.update_layout(
        margin=dict(t=20, l=25, r=25, b=25),
        showlegend=False,
        plot_bgcolor='white',
        height=400,
        width=650,
        colorway=color_range,
        coloraxis_showscale=False,
    )

    return fig22


# ## Heat Map - Figure Access_2 - callback for heatmap
@app.callback(
    Output('heatmap_plot', 'figure'),
    [Input('heatmap_plot', 'id')]
)
def update_heatmap(id):
    z = df_access2_1.values.astype(float)
    x = df_access2_1.columns.tolist()
    y = df_access2_1.index.tolist()

    z_percentage = np.round(z * 100, 2)

    colorscale = [[0, '#bee9e8'], [1, '#62b6cb']]
    fig_access2 = ff.create_annotated_heatmap(z=z_percentage, x=x, y=y, colorscale=colorscale)

    # Update annotations to display values as percentages
    for i in range(len(fig_access2.layout.annotations)):
        fig_access2.layout.annotations[i].text = f'{z_percentage[i // len(x), i % len(x)]:.0f}%'

    # Add labels and title
    fig_access2.update_layout(
        title='Percentage of respondents perceiving access to cash withdrawals as fairly difficult or very difficult, '
              'by country, 2022', font=dict(family='Tableau Bold', size=15),
        # xaxis=dict(title='Country'),
        yaxis=dict(  # title='Perceiving Difficulty',

            tickangle=-90,  # Rotate the y-axis labels by 90 degrees
            tickfont=dict(family='Tableau Bold', size=12),  # Set the font style for y-axis labels

        )
    )

    # fig_access2.update_annotations(dict(xref='paper', x=0.001), font_color='black', font_family='Tableau Book',
    # textangle=-90)

    return fig_access2


# End of code for heatmap - Access 2.1

# ## CALLBACK  ### END ###
##############################


# ################ Always finish the Dash app code with this line ###
if __name__ == '__main__':
    app.run_server(debug=True)

# ################  END OF THE DASH APP CODE ###
########################################################################################################################
