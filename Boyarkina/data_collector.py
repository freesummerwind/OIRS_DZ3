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
                 photos_number, photos_info, music_number, music_info):
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
        """
        self.__text = text
        self.__likes = likes_number
        self.__reposts = reposts_number
        self.__type = post_type
        self.__photos_num = photos_number
        self.__photos_info = photos_info
        self.__music_num = music_number
        self.__music_info = music_info

    def get_fields(self):
        return {
            'text': self.__text,
            'likes': self.__likes,
            'reposts': self.__reposts,
            'type': self.__type,
            'photos_number': self.__photos_num,
            'photos_info': self.__photos_info,
            'music_number': self.__music_num,
            'music_info': self.__music_info
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
            result_posts.append(Post(post['text'], post['likes']['count'], post['reposts']['count'], post['post_type'],
                                     photo_num, photos, music_num, music))
        result_groups.append(
            Group(-1 * group['id'], group['name'], group['status'], group['description'], group['type'],
                  group['activity'], group['can_post'], group['can_suggest'], group['main_section'],
                  group['members_count'], group['verified'], group['photo_100'], result_posts,
                  themes[i]))

    return result_groups
