from dash import Dash, html, dcc, Input, State, Output
from dash.exceptions import PreventUpdate
import json
import sqlite3
import vk_api

VK_TOKEN = 'token'
FILENAME = 'database.db'


class Group:
    """
    Класс, описывающий группу вк, ее фичи и тематику
    Класс неизменяемый, все параметры задаются при создании один раз. Для всех параметров написаны геттеры.
    Других функций у класса нет, нужен для более удобного представления группы
    """

    def __init__(self, group_id, name, status, description, group_type, activity, can_make_post, can_suggest_post,
                 main_section, subscribers_number, is_verified, photo_url, posts, theme):
        """
        Конструктор класса
        :param group_id: идентификатор группы, int
        :param name: название группы, string
        :param status: статус группы, string
        :param description: описание группы, string
        :param group_type: тип группы, string
        :param activity: базовая активность группы, string
        :param can_make_post: может ли пользователь создать пост, int
        :param can_suggest_post: может ли пользователь предложить пост, int
        :param main_section: главная секция группы, int
        :param subscribers_number: число подписчиков, int
        :param is_verified: верифицирована ли группа, int
        :param photo_url: ссылка на главное фото группы, string
        :param posts: информация о последних 15 записях группы, массив объектов класса Post
        :param theme: тематика группы, string
        """
        self.__id = group_id
        self.__name = name
        self.__status = status
        self.__desc = description
        self.__type = group_type
        self.__activity = activity
        self.__can_make_post = can_make_post
        self.__can_suggest_post = can_suggest_post
        self.__main_section = main_section
        self.__subscribers = subscribers_number
        self.__is_verified = is_verified
        self.__photo_url = photo_url
        self.__last_15_posts = posts
        self.__theme = theme

    def get_id(self):
        return self.__id

    def get_name(self):
        return self.__name

    def get_features(self):
        return [
            self.__status,
            self.__desc,
            self.__type,
            self.__activity,
            self.__can_make_post,
            self.__can_suggest_post,
            self.__main_section,
            self.__subscribers,
            self.__is_verified,
            self.__photo_url
        ]

    def get_posts(self):
        prepared_posts = []
        for post in self.__last_15_posts:
            prepared_posts.append(post.get_fields())
        return prepared_posts

    def get_theme(self):
        return self.__theme


class Post:
    """
    Класс, описывающий пост вк
    Класс неизменяемый, все параметры задаются при создании один раз. Написан единый геттер для всех параметров.
    Других функций у класса нет, нужен для более емкого и удобного представления поста
    """

    def __init__(self, text, likes_number, reposts_number, post_type,
                 photos_number, photos_info, music_number, music_info, video_number, video_info):
        """
        Конструктор класса
        :param text: текст поста, string
        :param likes_number: число лайков на посте, int
        :param reposts_number: число репостов этого поста, int
        :param post_type: тип поста, string
        :param photos_number: число фотографий, прикрепленных к посту; int
        :param photos_info: информация о фото, прикрепленных к посту. массив элементов вида
        {'text': 'text', 'url': 'url'}, где по ключу text доступна подпись к фото, а url содержит ссылку на это фото
        в среднем качестве
        :param music_number: число аудиозаписей, прикрепленных к посту; int
        :param music_info: информация об аудиозаписях, прикрепленных к посту. массив элементов вида
        {'title': 'title', 'artist': 'artist'}, где по ключу title доступно название трека, а artist - исполнитель трека
        :param video_number: число видеозаписей, прикрепленных к посту; int
        :param video_info: информация о видеозаписях, прикрепленных к посту. массив элементов вида
        {'title': 'title', 'duration': dur}, где по ключу title доступно название видео, а dur - продолжительность видео
        """
        self.__text = text
        self.__likes = likes_number
        self.__reposts = reposts_number
        self.__type = post_type
        self.__photos_num = photos_number
        self.__photos_info = photos_info
        self.__music_num = music_number
        self.__music_info = music_info
        self.__video_num = video_number
        self.__video_info = video_info

    def get_fields(self):
        return {
            'text': self.__text,
            'likes': self.__likes,
            'reposts': self.__reposts,
            'type': self.__type,
            'photos_number': self.__photos_num,
            'photos_info': self.__photos_info,
            'music_number': self.__music_num,
            'music_info': self.__music_info,
            'video_number': self.__video_num,
            'video_info': self.__video_info
        }


def get_groups_info(vk, groups_id, themes):
    """
    Функция, которая выгружает данные о группах вк
    :param vk: api вк сессии
    :param groups_id: id/address групп вк, массив строк
    :param themes: тематики соответствующих групп вк, массив строк
    :return: список групп с информацией о них, массив объектов класса Group
    """
    getting_fields = ['name', 'status', 'description', 'type', 'activity', 'can_post', 'can_suggest',
                      'main_section', 'members_count', 'verified', 'photo_100']
    groups_info = vk.groups.getById(group_ids=groups_id, fields=getting_fields)
    result_groups = []
    for i in range(len(groups_info)):
        group = groups_info[i]
        posts_info = vk.wall.get(owner_id=-1 * group['id'], count=15)
        result_posts = []
        for post in posts_info['items']:
            media = post['attachments']
            photo_num = 0
            photos = []
            music_num = 0
            music = []
            video_num = 0
            video = []
            for element in media:
                if element['type'] == 'photo':
                    photo_num += 1
                    photos.append({
                        'text': element['photo']['text'],
                        'url': element['photo']['sizes'][3]['url']
                    })
                elif element['type'] == 'audio':
                    music_num += 1
                    music.append({
                        'title': element['audio']['title'],
                        'artist': element['audio']['artist']
                    })
                elif element['type'] == 'video':
                    video_num += 1
                    video.append({
                        'title': element['video']['title'],
                        'duration': element['video']['duration']
                    })
            result_posts.append(Post(post['text'], post['likes']['count'], post['reposts']['count'], post['post_type'],
                                     photo_num, photos, music_num, music, video_num, video))
        result_groups.append(
            Group(-1 * group['id'], group['name'], group['status'], group['description'], group['type'],
                  group['activity'], group['can_post'], group['can_suggest'], group['main_section'],
                  group['members_count'], group['verified'], group['photo_100'], result_posts,
                  themes[i]))

    return result_groups


def parse_vk_groups(vk_token, groups_id, themes, db_path):
    """
    Функция, которая подключается к вк, получает данные о группах и записывает их в базу данных
    :param vk_token: вк токен, string
    :param groups_id: список адресов групп, массив string
    :param themes: тематики групп, массив string
    :param db_path: путь к файлу, содержащему sqlite базу данных, string
    :return: ничего не возвращает
    """
    vk_session = vk_api.VkApi(token=vk_token)
    vk = vk_session.get_api()
    parsed_groups = get_groups_info(vk, groups_id, themes)
    db = sqlite3.connect(db_path)
    cursor_obj = db.cursor()
    for group in parsed_groups:
        cursor_obj.execute("insert into groups values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [group.get_id(), group.get_name()] + group.get_features()
                           + [json.dumps(group.get_posts(), ensure_ascii=False)] + [group.get_theme()])
    db.commit()
    db.close()


def get_10_groups_from_db(db_path):
    """
    Функция, возвращающая html таблицу, содержащую информацию о 10 последних добавленных группах в базу данных
    :param db_path: путь к файлу, содержащему sqlite базу данных, string
    :return: html-таблица с 10 группами из базы данных
    """
    column_names = ['id', 'Название', 'Статус', 'Описание', 'Тип', 'Активность', 'Можно постить', 'Можно предлагать',
                    'Основная секция', 'подписчики', 'Верифицирована', 'Главное фото', 'Посты', 'Тематика']
    db = sqlite3.connect(db_path)
    cursor_obj = db.cursor()
    cursor_obj.execute("select * from groups")
    rows = cursor_obj.fetchall()
    db.close()
    rows.reverse()
    if len(rows) > 10:
        rows = rows[:10]
    result_table = []
    for row in rows:
        json_posts = json.loads(row[12])
        posts = ""
        for i in range(len(json_posts)):
            posts += f'Пост {i + 1}. Текст: "{json_posts[i]["text"]}". {json_posts[i]["photos_number"]} фото, ' + \
                     f'{json_posts[i]["video_number"]} видео, {json_posts[i]["music_number"]} аудио прикреплено к ' \
                     f'посту. {json_posts[i]["likes"]} лайков, {json_posts[i]["reposts"]} репостов.\n'
        row = list(row)
        row[11] = html.Img(src=row[11])
        row[12] = posts
        result_table.append(row)
    rows = result_table
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col, style={'border': '1px solid'}) for col in column_names])
        ),
        html.Tbody([
            html.Tr([
                html.Td(rows[row][col], style={'border': '1px solid'}) for col in range(len(rows[0]))
            ]) for row in range(len(rows))
        ])
    ], style={'border': '2px solid', 'border-collapse': 'collapse'})


"""
Далее представлен код, создающий веб-приложение по адресу http://127.0.0.1:8050/ для добавления групп в базу данных
"""
database = sqlite3.connect(FILENAME)
cursor = database.cursor()
cursor.execute("create table if not exists groups(id integer, name text, status text, "
               "description text, type text, activity text, can_post integer, can_suggest integer, "
               "main_section integer, subscribers integer, verified integer, main_photo text, posts json,"
               "theme text)")
database.commit()
database.close()
app = Dash(__name__)
app.layout = html.Div([
    html.H1(children='Домашнее задание 3 по ОИРС'),
    html.Div(children=[
        html.Label('адрес группы ВК (после vk.com/)'),
        html.Br(),
        dcc.Input(id='input_address', type='text')
    ]),
    html.Div(children=[
        html.Label('Тематика'),
        html.Br(),
        dcc.Dropdown(['Музыка', 'Путешествия', 'Программирование', 'Мемы', 'Другое'], 'Другое', id='input_theme')
    ]),
    html.Button(children=['Сохранить'], id='save_btn'),
    html.Br(),
    html.H3(children='', style={'color': 'red'}, id='error_msg'),
    html.Br(),
    html.Div(children=[get_10_groups_from_db(FILENAME)], id='accounts_table')
])


@app.callback(
    Output('accounts_table', 'children'),
    Output('error_msg', 'children'),
    Input('save_btn', 'n_clicks'),
    State('input_address', 'value'),
    State('input_theme', 'value')
)
def update_table(n_clicks, input_address, input_theme):
    if n_clicks is None:
        raise PreventUpdate
    if input_address is None:
        return [get_10_groups_from_db(FILENAME)], 'Введите адрес группы'
    if input_theme is None:
        return [get_10_groups_from_db(FILENAME)], 'Выберите тематику группы'
    try:
        parse_vk_groups(VK_TOKEN, [input_address], [input_theme], FILENAME)
    except:
        return [get_10_groups_from_db(FILENAME)], 'Частная группа, невозможно получить информацию'
    return [get_10_groups_from_db(FILENAME)], ''


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')
