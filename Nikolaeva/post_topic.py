#!/usr/bin/python
# -*- coding: utf-8 -*-

from dash import Dash, html, dcc, dash_table, Patch
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pypyodbc
import sys
import numpy as np
import pickle
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import codecs
from sklearn import linear_model
import vk_api

myServer = 'DESKTOP-UJ2J63A\SQLEXPRESS'
myDatabase = 'VK_POSTS'
access_token = ''

def get_group_name(vk_api, id_gr):
    name = vk.groups.getById(group_id=(id_gr))[0]
    name = name['name']
    return name

def get_group_activity(vk_api, id_gr):
    activity = vk.groups.getById(group_id=(id_gr), fields=['activity'])[0]
    if 'activity' in activity:
        return activity['activity']
    return None

def get_group_status(vk_api, id_gr):
    status = vk.groups.getById(group_id=(id_gr), fields=['status'])[0]
    if 'status' in status:
        return status['status']
    return None

def get_group_description(vk_api, id_gr):
    descr = vk.groups.getById(group_id=(id_gr), fields=['description'])[0]
    if 'description' in descr:
        return descr['description']
    return None

def get_post_text(vk_api, id_post):
    post = vk.wall.getById(posts=id_post)[0]
    if 'text' in post:
        text = post['text']
        return text
    return None

def get_post_photo(vk_api, id_post):
    post = vk.wall.getById(posts=id_post)[0]
    if 'attachments' not in post:
        return None
    attachments = post['attachments']
    photo = None
    for i in range(len(attachments)):
        if attachments[i]['type'] == 'photo':
            photo = attachments[i]['photo']['sizes'][0]['url']
            break
    return photo

def get_topics(vk_api, id_gr):
    topics = vk.board.getTopics(group_id=id_gr)
    if 'items' not in topics:
        return None
    topics = topics['items']
    group_topics = ''
    for i in range(len(topics)):
        if i == 7: break
        group_topics = group_topics + topics[i]['title'] + ', '
    return group_topics

def get_group_avatar(vk_api, id_gr):
    avatar = vk.groups.getById(group_id=id_gr, fields='photo_50')[0]
    avatar = avatar['photo_50']
    return avatar

def get_group_posts(vk_api, id_gr):
    posts = vk.wall.get(owner_id='-' + id_gr)
    posts = posts['items']
    posts_text = ''
    for i in range(len(posts)):
        if i == 7: break
        post = posts[i]
        if 'text' not in post: continue
        posts_text = posts_text + post['text'] + '; '
    return posts_text

def get_followers(vk_api, id_gr):
    followers = vk.groups.getById(group_id=id_gr, fields='members_count')[0]
    followers = followers['members_count']
    return followers

def get_media(vk_api, id_gr):
    videos = 0
    photos = 0
    articles = 0
    audios = 0
    media = vk.groups.getById(group_id=id_gr, fields='counters')[0]
    media = media['counters']
    if 'videos' in media:
        videos = media['videos']
    if 'photos' in media:
        photos = media['photos']
    if 'audios' in media:
        audios = media['audios']
    if 'articles' in media:
        articles = media['articles']
    return photos, videos, audios, articles


def get_post_info (vk_api, id_post):
    info = []
    id_post = '-' + id_post
    id_gr = id_post[1:].split('_')[0]
    info.append(id_post)
    info.append(get_group_name(vk_api, id_gr))
    info.append(get_group_activity(vk_api, id_gr))
    info.append(get_group_status(vk_api, id_gr))
    info.append(get_group_description(vk_api, id_gr))
    info.append(get_post_text(vk_api, id_post))
    info.append(get_post_photo(vk_api, id_post))
    info.append(get_topics(vk_api, id_gr))
    info.append(get_group_avatar(vk_api, id_gr))
    info.append(get_group_posts(vk_api, id_gr))
    info.append(get_followers(vk_api, id_gr))
    info.append(get_media(vk_api, id_gr)[0])
    info.append(get_media(vk_api, id_gr)[1])
    info.append(get_media(vk_api, id_gr)[2])
    info.append(get_media(vk_api, id_gr)[3])
    return info

def get_post_vec(vk_api, z, model):
    your_post = get_post_info(vk_api, z)
    post_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range (1, len(your_post)):
        if (i==6) or (i==8): continue
        if your_post[i] == '':
            post_vec[i-1] = 0
        else:
            zz=[]
            zz.append(z)
            if model == 'sgd':
                predicted = sgd_clf.predict(zz)
            else:
                predicted = knb_ppl_clf.predict(zz)
            post_vec[i-1] = class_to_num(predicted[0])
        if i>=10:
            post_vec[i-1] = your_post[i]
    return post_vec

def text_cleaner(text):
    text = text.lower()
    reg = re.compile('[^а-яА-Я ]')
    res = reg.sub('', text)
    for i in range(len(res)):
        if text[i] == ' ': res = res[i+1:]
        if text[i] != ' ': break
    return  res

def class_to_num(name):
    if name == 'Кулинария':
        return 1
    if name == 'Наука':
        return 2
    if name == 'Путешествия':
        return 3
    if name == 'Спорт':
        return 4
    if name == 'Другое':
        return 5

# загружаем данные для обучения модели
def load_data():
    data = {'text':[],'tag':[]}
    for j in range(len(loading)):
        for i in range (len(loading[j]) - 1):
            data['text'] += [text_cleaner(loading[j][i])]
            data['tag'] += [loading[j][-1]]
    return data


def train_test_split(data, validation_split = 0.05):
    n = len(data['text'])
    indices = np.arange(n)
    np.random.shuffle(indices)

    X = [data['text'][i] for i in indices]
    Y = [data['tag'][i] for i in indices]
    nb_validation_samples = int( validation_split * n )

    return {
        'train': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]},
        'test': {'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]}
    }

connection = pypyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                                  'Server=' + myServer + ';'
                                  'Database=' + myDatabase + ';'
                                  'Trusted_Connection=yes;')
cursor = connection.cursor()
mySQLQuery = ("""
                    SELECT * FROM dbo.posts 
              """)

cursor.execute(mySQLQuery)
result = cursor.fetchall()
myData = []
for post in result:
    temp = []
    for info in post:
        temp.append(info)
    myData.append(temp)
cursor.close()
connection.close()


# формируем вектор значений номеров классив для каждого поста
n = len(myData)
myData_vec = []
for i in range (n):
    myData_vec.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
loading = []


for i in range(len(myData)):
    temp = []
    for k in range (1, 9):
        if (k==6) or (k==8): continue
        if myData[i][k] == '':
            myData_vec[i][k-1] = 0
        else:
            myData_vec[i][k-1] = class_to_num(myData[i][15])
            temp.append(myData[i][k])
    temp.append(myData[i][15])
    loading.append(temp)
    for j in range (9,14):
        myData_vec[i][j] = myData[i][j+1]

# обучаем модель SGDClassifier
data = load_data()
D = train_test_split(data)
sgd_clf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), strip_accents = 'unicode')),
    ('clf', SGDClassifier(loss = 'hinge', penalty= 'elasticnet', class_weight= None))])
sgd_clf.fit(D['train']['x'], D['train']['y'])

knb_ppl_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('knb_clf', KNeighborsClassifier(n_neighbors=10))])
knb_ppl_clf.fit(D['train']['x'], D['train']['y'])

features = myData_vec
labels = []
for i in range (len(myData)):
    labels.append(myData[i][15])

reg = linear_model.RidgeClassifier()
reg.fit(features, labels)


vk_session = vk_api.VkApi(token=access_token)
vk = vk_session.get_api()

# создание интерфейса
app = Dash(__name__)

app.layout = html.Div([
    html.H4("Введите id поста"),
    dcc.Input(id='input-1-state', type='text', value=''),
    html.H4("Выберите модель"),
    dcc.Dropdown(['SGDClassifier', 'KNEibour'], multi=False, id = 'input-2-state'),
    html.H4("Возможные тематики: Кулинария, Наука, Путешествия, Спорт, Другое"),
    html.Button(id='submit-button-state', children='Узнать тематику поста'),
    html.H4("Тематика вашего поста"),
    dash_table.DataTable(
            data = [ {'id': '', 'Тематика': ''
                      }],
            page_size=10,
            id="mytable",
        )
])

# обработка нажатия кнопки
@app.callback(Output('mytable', 'data'),
              Input('submit-button-state', 'n_clicks'),
              State('input-1-state', 'value'),
              State('input-2-state', 'value'))

def update_output(n_clicks, input1, input2):
    if n_clicks is None:
        raise PreventUpdate
    else:
        post_vec = get_post_vec(vk_api, input1, input2)

        result = reg.predict([post_vec])
        patched_table = Patch()
        patched_table.extend([{'id': input1, 'Тематика': result[0]}])

        if (n_clicks > 10) or (n_clicks) == 1:
            del patched_table[0]
        return patched_table

if __name__ == '__main__':
    app.run_server(debug=True)


