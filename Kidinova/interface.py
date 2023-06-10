from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
from dash import dash_table
import plotly.figure_factory as ff
import sqlite3

from config import path_to_sqlite_db
from database import get_last_rows, insert_data
from parser import parse


def save_dataset(path, df):
    df.to_csv(path, index=False)


connection = sqlite3.connect(path_to_sqlite_db)
cursor = connection.cursor()
group_df = get_last_rows(connection, 'vk_group_info')
post_df = get_last_rows(connection, 'vk_post_info')

group_fig = ff.create_table(group_df)
post_fig = ff.create_table(post_df)

app = Dash(__name__)


app.layout = html.Div(
    children=[
        html.H1(children='Parsing group', style={'textAlign': 'center'}),
        html.Div(
            children=[
                dcc.Input(id='input_url', placeholder='Enter URL here...',
                          value=None, type='text',
                          style={'width': '70%', 'textAlign': 'center', 'fontSize': '15px'}),
                html.Button('Parse group', id='button', n_clicks=0,
                            style={'width': '30%', 'height': '40px', 'fontSize': '15px'})
            ],
            style=dict(display='flex', justifyContent='center')),
        html.H2(children='Group dataset', style={'textAlign': 'center'}),
        html.Div(
            children=[
                dash_table.DataTable(
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'lineHeight': '15px'
                    },
                    id='table_group',
                    columns=[{
                        'name': group_df.columns[i],
                        'id': group_df.columns[i]
                    } for i in range(len(group_df.columns))],
                    data=[
                        {group_df.columns[i]: group_df.values[j][i] for i in range(len(group_df.columns))}
                        for j in range(-10, 0)
                    ],
                    style_table={'overflowX': 'scroll'},
                )
            ],
            style={'width': '100%'}
        ),
        html.H2(children='Post dataset', style={'textAlign': 'center'}),
        html.Div(
            children=[
                dash_table.DataTable(
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'lineHeight': '15px'
                    },
                    id='table_post',
                    columns=[{
                        'name': post_df.columns[i],
                        'id': post_df.columns[i]
                    } for i in range(len(post_df.columns))],
                    data=[
                        {post_df.columns[i]: post_df.values[j][i] for i in range(len(post_df.columns))}
                        for j in range(-10, 0)
                    ],
                    style_table={'overflowX': 'scroll'},
                )
            ],
            style={'width': '100%'}
        )
    ],
)


@callback(
    [Output('table_group', 'data'), Output('table_post', 'data'), Output('input_url', 'value')],
    [Input('button', 'n_clicks')],
    [State('input_url', 'value')]
)
def update_table(clicks, input_url):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    con = sqlite3.connect(path_to_sqlite_db)
    group_df_ = get_last_rows(con, 'vk_group_info')
    post_df_ = get_last_rows(con, 'vk_post_info')
    if 'button' in changed_id:
        print(clicks, input_url)
        try:
            group_info_df, group_posts_df = parse(input_url)
            print(group_info_df.head())
            insert_data(con, group_info_df, group_posts_df)
            group_df_ = get_last_rows(con, 'vk_group_info')
            post_df_ = get_last_rows(con, 'vk_post_info')
        except Exception as e:
            print(f'Error while handling {input_url}: \n\t{e}')

    return [{group_df.columns[i]: group_df_.values[j][i] for i in range(len(group_df_.columns))}
            for j in range(-10, 0)], \
        [{post_df.columns[i]: post_df_.values[j][i] for i in range(len(post_df_.columns))}
            for j in range(-10, 0)], ''


if __name__ == '__main__':
    app.run_server(debug=True)
