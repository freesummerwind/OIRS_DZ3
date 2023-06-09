import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import tensorflow as tf
import requests
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
import pickle

model_path = '/Users/kirillshevchenko/Desktop/8сем/оирс/3 дз/hw3/path_to_my_model.h5'
tokenizer_path = '/Users/kirillshevchenko/Desktop/8сем/оирс/3 дз/hw3/tokenizer.pickle'
label_encoder_path = '/Users/kirillshevchenko/Desktop/8сем/оирс/3 дз/hw3/label_encoder.pickle'

model = tf.keras.models.load_model(model_path)

# Load tokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Load LabelEncoder
with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)


app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/carbon-components@10.0.0/carbon-components.min.css'])

app.layout = html.Div(
    style={"display": "flex", "flex-direction": "column", "align-items": "center"},
    children=[
        html.H1('Предсказание темы постов в группе ВКонтакте'),

        html.Div(
            style={"margin-bottom": "20px"},
            children=[
                dcc.Input(
                    id='group-url',
                    type='text',
                    placeholder='Введите URL группы...',
                    style={"width": "400px", "height": "40px", "font-size": "18px"}
                )
            ]
        ),

        html.Div(
            style={"margin-bottom": "20px"},
            children=[
                dcc.Input(
                    id='num-predictions',
                    type='number',
                    placeholder='Количество предсказаний...',
                    style={"width": "250px", "height": "40px", "font-size": "18px"}
                )
            ]
        ),

        html.Button('Предсказать', id='predict-button', n_clicks=0),

        html.Div(id='prediction-result')
    ]
)

# Определение max_sequence_length
max_sequence_length = 100

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

def get_posts_from_vk(group_link, count):
    access_token = ''
    version = '5.130'
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
        posts = [item['text'] for item in data['response']['items']]
        return posts
    else:
        print('Error:', data['error']['error_msg'])
        return None


def predict_post_theme(post):
    sequences = tokenizer.texts_to_sequences([post])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequences)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction, axis=1)])
    return predicted_label[0]

# CSS стили для таблицы
table_style = {
    'border': '1px solid black',
    'border-collapse': 'collapse',
    'width': '100%'
}

# CSS стили для ячеек таблицы
table_cell_style = {
    'border': '1px solid black',
    'padding': '8px'
}

@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-button', 'n_clicks'),
    State('group-url', 'value'),
    State('num-predictions', 'value')
)
def update_output(n_clicks, value, num_predictions):
    if n_clicks > 0:
        if value and num_predictions:
            num_predictions = int(num_predictions)
            group_posts = get_posts_from_vk(value, num_predictions)
            if group_posts:
                predictions = [predict_post_theme(post) for post in group_posts]
                df = pd.DataFrame({'Посты группы': group_posts, 'Предсказания': predictions})
                table = html.Table(
                    className='table',
                    style=table_style,  # Добавлено применение стилей к таблице
                    children=[
                        html.Thead(
                            html.Tr([
                                html.Th('Посты группы', className='table-cell', style=table_cell_style),  # Добавлено применение стилей к заголовкам
                                html.Th('Предсказания', className='table-cell', style=table_cell_style)  # Добавлено применение стилей к заголовкам
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(df.iloc[i]['Посты группы'], className='table-cell', style=table_cell_style),  # Добавлено применение стилей к ячейкам данных
                                html.Td(df.iloc[i]['Предсказания'], className='table-cell', style=table_cell_style)  # Добавлено применение стилей к ячейкам данных
                            ]) for i in range(len(df))
                        ])
                    ]
                )
                return table
            else:
                return 'В этой группе нет постов.'
        else:
            return 'Пожалуйста, введите URL группы и выберите количество предсказаний.'


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)
