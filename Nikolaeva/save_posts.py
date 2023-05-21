#!/usr/bin/python
# -*- coding: utf-8 -*-

from dash import Dash, html, dcc, dash_table, Patch
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import vk_api
import pypyodbc

'''
Будем классифицировать посты по 4 классам (темам):
1) Кулинария
2) Наука
3) Путешествия
4) Спорт
5) Другое

Используем следующие признаки:
1) Название сообщества, в котором опубликован пост
2) Тематика сообщества
3) Статус сообщества
4) Описание сообщества
5) Текст в посте
6) Первое изображение в посте
7) Названия первых 7 обсуждений в группе
8) Аватарка группы
9) Текст в последних 7 записях в сообществе
10) Количество подписчиков сообщества
11) Количество фото в сообществе
12) Количество видео в сообществе
13) Количество аудиозаписей в сообществе
14) Количество статей в сообществе
'''

myServer = 'DESKTOP-UJ2J63A\SQLEXPRESS'
myDatabase = 'VK_POSTS'

access_token = 'vk1.a._ARQSgjFy7fBFSBXb_GJAO3hKWQ4U3KyU2f8x6XG83jRg94yv8m1Ivd5z_OjjMsmo9xPTHS4hXyE-RhWnb4osGeYp9zgO0S4BXZ85wzNsnPGunUNcDGtPsJ_QhdtE0TZc0gG7QB0LF8KsZoBwchyqqYrJULst8UtKz16nxKBr0joStqJtHfFu_QlUIkOT_FzVUwmK0BZlIsxlFuIug_xEw'
vk_session = vk_api.VkApi(token=access_token)
vk = vk_session.get_api()


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

# запись в базу данных
def write_to_db(myData):
    connection = pypyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                                  'Server=' + myServer + ';'
                                  'Database=' + myDatabase + ';'
                                  'Trusted_Connection=yes;')
    cursor = connection.cursor()
    mySQLQuery = ("""
                    INSERT INTO dbo.posts (id, name, activity, status, description, text, image, topics,
                    avatar, posts, followers, photos, videos, audios, articles, class)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """)
    data_tuple = (myData[0], myData[1], myData[2], myData[3], myData[4],
                  myData[5], myData[6], myData[7], myData[8], myData[9],
                  myData[10], myData[11], myData[12], myData[13], myData[14], myData[15])
    cursor.execute(mySQLQuery, data_tuple)
    connection.commit()
    print("Данные успешно вставлены в таблицу")
    cursor.close()
    connection.close()


# создание интерфейса
app = Dash(__name__)

app.layout = html.Div([
    html.H4("Введите id поста"),
    dcc.Input(id='input-1-state', type='text', value=''),
    html.H4("Выберите тематику"),
    dcc.Dropdown(['Кулинария', 'Наука', 'Путешествия', 'Спорт', 'Другое'], ['Спорт'], multi=False, id = 'input-2-state'),
    html.Button(id='submit-button-state', children='Сохранить'),
    html.H4("Сохраненные посты"),
    dash_table.DataTable(
            data = [ {'id': '', 'название сообщества': '', 'тематика сообщества': '',
                      'статус сообщества': '', 'описание сообщества': '',
                      'текст в посте': '', 'изображения в посте': '', 'обсуждения': '',
                      'аватар': '', 'текст в последних записях': '', 'подписчики': '',
                      'фото': '', 'видео': '', 'аудио': '', 'статьи': '', 'класс': ''
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
        id_post = '-' + input1
        id_gr = id_post[1:].split('_')[0]
        result = input2
        myData = []
        myData.append(input1)

        name = get_group_name(vk_api, id_gr)
        myData.append(name)
        activity = get_group_activity(vk_api, id_gr)
        myData.append(activity)
        status = get_group_status(vk_api, id_gr)
        myData.append(status)
        descr = get_group_description(vk_api, id_gr)
        myData.append(descr)

        text = get_post_text(vk_api, id_post)
        myData.append(text)
        if len(text)>15:
            text = text[:15]+'...'
        if text is None:
            text = '-'

        image = get_post_photo(vk_api, id_post)
        myData.append(image)
        if image is None:
            image = 'нет'
        else:
            image = 'да'

        topics = get_topics(vk_api, id_gr)
        myData.append(topics)
        if (topics == None) or (len(topics) == 0):
            topics = '-'
        else:
            if len(topics) > 15:
                topics = topics[:15]+'...'

        avatar = get_group_avatar(vk_api, id_gr)
        myData.append(avatar)
        if len(avatar) > 15:
            avatar = avatar[:15]+'...'

        last_posts = get_group_posts(vk_api, id_gr)
        myData.append(last_posts)
        if (last_posts is None) or (len(last_posts) == 0):
            last_posts = '-'
        else:
            if len(last_posts) > 15:
                last_posts = last_posts[:15]+'...'

        followers = get_followers(vk_api, id_gr)
        myData.append(followers)
        media = get_media(vk_api,id_gr)
        myData.append(media[0])
        myData.append(media[1])
        myData.append(media[2])
        myData.append(media[3])

        myData.append(input2)
        write_to_db(myData)

        patched_table = Patch()
        patched_table.extend([{'id': id_post[1:], 'название сообщества': name, 'тематика сообщества': activity,
                      'статус сообщества': status, 'описание сообщества': descr,
                      'текст в посте': text, 'изображения в посте': image,
                      'обсуждения': topics, 'аватар': avatar, 'текст в последних записях': last_posts,
                      'подписчики': followers, 'фото': media[0],
                      'видео': media[1], 'аудио': media[2], 'статьи': media[3], 'класс': result,
                      }])

        if (n_clicks > 10) or (n_clicks) == 1:
            del patched_table[0]
        return patched_table

if __name__ == '__main__':
    app.run_server(debug=True)



