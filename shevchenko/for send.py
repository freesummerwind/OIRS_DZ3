import requests
import json
import psycopg2
import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

def get_group_id_from_link(group_link, access_token='', version='5.130'):
    screen_name = group_link.split('/')[-1]
    url = 'https://api.vk.com/method/utils.resolveScreenName'
    params = {
        'screen_name': screen_name,
        'access_token': access_token,
        'v': version
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'response' in data and data['response']['type'] == 'group':
        return -data['response']['object_id']
    else:
        print("Invalid group link format")
        return None

def get_posts_from_vk(group_link, access_token='', version='5.130', count=100):
    group_id = get_group_id_from_link(group_link, access_token, version)
    if group_id is None:
        print("Unable to retrieve posts: Invalid group link")
        return []
    url = 'https://api.vk.com/method/wall.get'
    params = {
        'owner_id': group_id,
        'access_token': access_token,
        'v': version,
        'count': count
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'response' in data:
        posts = data['response']['items']
        return posts
    else:
        print('Error:', data['error']['error_msg'])
        return None


def save_posts_to_db(posts):
    conn = psycopg2.connect(
        dbname='oirs_dz3', 
        user='kirillshevchenko', 
        password='root', 
        host='localhost'
    )
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS vk_posts (
            id SERIAL PRIMARY KEY,
            post_id INT,
            text TEXT,
            likes INT,
            reposts INT,
            views INT
        )
    ''')

    for post in posts:
        cur.execute('''
            INSERT INTO vk_posts (post_id, text, likes, reposts, views) 
            VALUES (%s, %s, %s, %s, %s)
        ''', (
            post['id'], 
            post['text'], 
            post['likes']['count'], 
            post['reposts']['count'], 
            post['views']['count']
        ))

    conn.commit()
    cur.close()
    conn.close()

access_token = ''
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='group-link-input', type='text', placeholder='Введите ссылку на группу ВКонтакте'),
    html.Button(id='submit-button', n_clicks=0, children='Получить данные'),
    dash_table.DataTable(id='table')
])

@app.callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    [Input('submit-button', 'n_clicks')],
    [State('group-link-input', 'value')]
)


def update_table(n_clicks, group_link):
    if n_clicks > 0:
        try:
            posts = get_posts_from_vk(group_link, access_token)
            if posts:
                save_posts_to_db(posts)

                conn = psycopg2.connect(dbname='oirs_dz3', user='kirillshevchenko', password='root', host='localhost')
                df = pd.read_sql("SELECT * FROM vk_posts", conn)
                conn.close()

                data = df.to_dict('records')
                columns = [{"name": i, "id": i} for i in df.columns]
                return data, columns
        except psycopg2.Error as e:
            print(f"An error occurred with the database: {e}")
            return [], []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return [], []
    else:
        return [], []


if __name__ == '__main__':
    app.run_server(debug=True)
