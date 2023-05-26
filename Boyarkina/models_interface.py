from dash import Dash, html, dcc, Input, State, Output
from dash.exceptions import PreventUpdate
from gensim.models import Word2Vec
import joblib
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import re
from tensorflow import keras
import vk_api

from data_collector import get_groups_info, VK_TOKEN


patterns = "[«»!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru_en = stopwords.words("russian") + stopwords.words("english")
morph = MorphAnalyzer()


def lemmatize(doc):
    doc = doc.lower()
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        token = token.strip()
        if token and token not in stopwords_ru_en:
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None


def get_text_similarity(tokenized_text, theme_word):
    w2v_model = Word2Vec.load("models/word2vec.model")
    text_similarity = 0.
    used_words = 0
    for word in tokenized_text:
        if word in w2v_model.wv.key_to_index:
            text_similarity += w2v_model.wv.similarity(word, theme_word)
            used_words += 1
    if used_words != 0:
        return text_similarity / used_words
    return 0.


def get_themes_characteristics(tokenized_text):
    return [
        get_text_similarity(tokenized_text, 'музыка'),
        get_text_similarity(tokenized_text, 'путешествие'),
        get_text_similarity(tokenized_text, 'программирование'),
        get_text_similarity(tokenized_text, 'мем')
    ]


def prepare_group_info_full_model(group_info):
    group_vector = []
    if lemmatize(group_info[1]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[1])))
    else:
        group_vector.extend([0.] * 4)
    if lemmatize(group_info[2]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[2])))
    else:
        group_vector.extend([0.] * 4)
    if lemmatize(group_info[3]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[3])))
    else:
        group_vector.extend([0.] * 4)
    group_vector.extend(group_info[6:11])
    for post in group_info[12]:
        if lemmatize(post['text']) is not None:
            group_vector.extend(get_themes_characteristics(lemmatize(post['text'])))
        else:
            group_vector.extend([0.] * 4)
        group_vector.extend([post['likes'], post['reposts'], post['photos_number'], post['music_number'],
                             post['video_number'], post['links_number'], post['docs_number']])
    while len(group_vector) < 3 * 4 + 5 + 15 * 11:
        group_vector.extend([0.] * 11)

    return group_vector


def prepare_group_info_no_posts_model(group_info):
    group_vector = []
    if lemmatize(group_info[1]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[1])))
    else:
        group_vector.extend([0.] * 4)
    if lemmatize(group_info[2]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[2])))
    else:
        group_vector.extend([0.] * 4)
    if lemmatize(group_info[3]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[3])))
    else:
        group_vector.extend([0.] * 4)
    group_vector.extend(group_info[6:11])

    return group_vector


def prepare_group_info_only_texts_model(group_info):
    group_vector = []
    if lemmatize(group_info[1]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[1])))
    else:
        group_vector.extend([0.] * 4)
    if lemmatize(group_info[2]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[2])))
    else:
        group_vector.extend([0.] * 4)
    if lemmatize(group_info[3]) is not None:
        group_vector.extend(get_themes_characteristics(lemmatize(group_info[3])))
    else:
        group_vector.extend([0.] * 4)
    for post in group_info[12]:
        if lemmatize(post['text']) is not None:
            group_vector.extend(get_themes_characteristics(lemmatize(post['text'])))
        else:
            group_vector.extend([0.] * 4)
    while len(group_vector) < 3 * 4 + 15 * 4:
        group_vector.extend([0.] * 11)

    return group_vector


themes_inverse = {
    1: 'музыка',
    2: 'путешествия',
    3: 'программирование',
    4: 'мемы',
    0: 'другое'
}


def get_group_theme(vk_token, group_address, model_number):
    vk_session = vk_api.VkApi(token=vk_token)
    vk = vk_session.get_api()
    parsed_group = get_groups_info(vk, [group_address], [''])[0]
    list_group = [
        parsed_group.get_id(),
        parsed_group.get_name()
    ]
    list_group.extend(parsed_group.get_features())
    list_group.append(parsed_group.get_posts())
    if model_number == 0:
        group_features = prepare_group_info_full_model(list_group)
        model = joblib.load('models/model_ridge_full.pkl')
        return 'Группа: ' + list_group[1] + ', тема поста: ' + themes_inverse[model.predict([group_features])[0]]
    elif model_number == 1:
        group_features = prepare_group_info_no_posts_model(list_group)
        model = joblib.load('models/model_ridge_no_posts.pkl')
        return 'Группа: ' + list_group[1] + ', тема поста: ' + themes_inverse[model.predict([group_features])[0]]
    elif model_number == 2:
        group_features = prepare_group_info_only_texts_model(list_group)
        model = joblib.load('models/model_ridge_only_texts.pkl')
        return 'Группа: ' + list_group[1] + ', тема поста: ' + themes_inverse[model.predict([group_features])[0]]
    elif model_number == 3:
        group_features = prepare_group_info_full_model(list_group)
        model = keras.models.load_model('models/model_seq_full.h5')
        prediction = model.predict([group_features])[0]
        max_index = 0
        for i in range(5):
            if prediction[i] > prediction[max_index]:
                max_index = i
        return 'Группа: ' + list_group[1] + ', тема поста: ' + themes_inverse[max_index]
    elif model_number == 4:
        group_features = prepare_group_info_no_posts_model(list_group)
        model = keras.models.load_model('models/model_seq_no_posts.h5')
        prediction = model.predict([group_features])[0]
        max_index = 0
        for i in range(5):
            if prediction[i] > prediction[max_index]:
                max_index = i
        return 'Группа: ' + list_group[1] + ', тема поста: ' + themes_inverse[max_index]
    else:
        group_features = prepare_group_info_only_texts_model(list_group)
        model = keras.models.load_model('models/model_seq_only_texts.h5')
        prediction = model.predict([group_features])[0]
        max_index = 0
        for i in range(5):
            if prediction[i] > prediction[max_index]:
                max_index = i
        return 'Группа: ' + list_group[1] + ', тема поста: ' + themes_inverse[max_index]


"""
Далее представлен код, создающий веб-приложение по адресу http://127.0.0.1:8050/ для работы с моделями
"""
app = Dash(__name__)
app.layout = html.Div([
    html.H1(children='Домашнее задание 3 по ОИРС'),
    html.H3(children='Интерфейс для работы с моделями'),
    html.Div(children=[
        html.Label('адрес группы ВК (после vk.com/)'),
        html.Br(),
        dcc.Input(id='input_address', type='text')
    ]),
    html.Div(children=[
        html.Label('Модель'),
        html.Br(),
        dcc.RadioItems(options=[
            {'label': 'Линейный классификатор, использующий все признаки группы', 'value': 0},
            {'label': 'Линейный классификатор, не использующий посты группы', 'value': 1},
            {'label': 'Линейный классификатор, использующий только текстовые признаки группы', 'value': 2},
            {'label': 'Многослойный перцептрон, использующий все признаки группы', 'value': 3},
            {'label': 'Многослойный перцептрон, не использующий посты группы', 'value': 4},
            {'label': 'Многослойный перцептрон, использующий только текстовые признаки группы', 'value': 5}
        ], id='input_model')
    ]),
    html.Button(children=['Предсказать тематику следующего поста'], id='predict_btn'),
    html.Br(),
    html.H4(children='', id='predicted_result'),
    html.Br(),
    html.H3(children='', style={'color': 'red'}, id='error_msg')
])


@app.callback(
    Output('predicted_result', 'children'),
    Output('error_msg', 'children'),
    Input('predict_btn', 'n_clicks'),
    State('input_address', 'value'),
    State('input_model', 'value')
)
def get_theme(n_clicks, input_address, model_number):
    if n_clicks is None:
        raise PreventUpdate
    if input_address is None:
        return '', 'Введите адрес группы'
    if model_number is None:
        return '', 'Выберите модель'
    try:
        group_theme = get_group_theme(VK_TOKEN, input_address, model_number)
    except:
        return '', 'Частная группа, невозможно получить информацию'
    return group_theme, ''


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')
