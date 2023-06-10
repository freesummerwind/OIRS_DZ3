from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
import time
from dash import dash_table
import pandas as pd
from predictor import get_posts, get_predict_cluster, get_predict_rnn


app = Dash(__name__)
group_df = pd.DataFrame({'owner_id': ['id of group'], 'text': ['some post']})

app.layout = html.Div(
    children=[
        html.H1(children='Predict next post', style={'textAlign': 'center'}),
        html.Div(
            children=[
                dcc.Input(id='input_url', placeholder='Enter URL to group here...',
                          value=None, type='text',
                          style={'width': '40%', 'textAlign': 'center', 'fontSize': '15px'}),
                html.Button('Predict', id='button', n_clicks=0,
                            style={'width': '20%', 'height': '40px', 'fontSize': '15px'})
            ],
            style=dict(display='flex', justifyContent='center')
        ),
        html.H2(children='Words from cluster', style={'textAlign': 'center'}),
        html.Div(
            children=[
                dcc.Textarea(
                    id='textarea-1',
                    value='Textarea content initialized\nwith multiple lines of text',
                    style={'width': '60%', 'height': 100, 'fontSize': '20px'},
                ),
            ],
            style=dict(display='flex', justifyContent='center')
        ),
        html.H2(children='Words from RNN encoder-decoder', style={'textAlign': 'center'}),
        html.Div(
            children=[
                dcc.Textarea(
                    id='textarea-2',
                    value=f'Here you will see predict of the model',
                    style={'width': '60%', 'height': 100, 'fontSize': '20px'},
                ),
            ],
            style=dict(display='flex', justifyContent='center')
        ),
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
                    data=group_df.to_dict('records'),
                    style_table={'overflowX': 'scroll'},
                )
            ],
            style={'width': '100%', 'justifyContent': 'center'}
        ),
    ],
)


@callback(
    [Output('textarea-1', 'value'), Output('textarea-2', 'value'),
     Output('input_url', 'value'), Output('table_group', 'data')],
    [Input('button', 'n_clicks')],
    [State('input_url', 'value')]
)
def update_table(clicks, input_url):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'button' in changed_id:
        print('button pushed value', input_url)
        group_df_, input_posts = get_posts(input_url)
        print('Input posts: ', input_posts)
        if input_posts is None:
            return '', '', '', group_df_.to_dict('records')

        text1 = get_predict_cluster(input_posts)
        print('text1', text1)
        # text2 = ''
        text2 = get_predict_rnn(input_posts)

        return text1, text2, '', group_df_.to_dict('records')
    return '', '', '', group_df.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)

