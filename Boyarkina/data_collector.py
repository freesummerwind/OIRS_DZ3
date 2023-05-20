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
        return {
            'status': self.__status,
            'description': self.__desc,
            'type': self.__type,
            'activity': self.__activity,
            'can_make_post': self.__can_make_post,
            'can_suggest_post': self.__can_suggest_post,
            'main_section': self.__main_section,
            'subscribers_number': self.__subscribers,
            'is_verified': self.__is_verified,
            'photo_url': self.__photo_url,
            'posts_15': self.__last_15_posts
        }

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
