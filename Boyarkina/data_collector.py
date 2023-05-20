class Group:
    """
    Класс, описывающий группу вк, ее фичи и тематику
    Класс неизменяемый, все параметры задаются при создании один раз. Для всех параметров написаны геттеры.
    Других функций у класса нет, нужен для более удобного представления группы
    """

    def __init__(self, group_id, name, features, theme):
        """
        Конструктор класса
        :param group_id: идентификатор группы, int
        :param name: название группы, string
        :param features: собранные фичи для группы, массив упорядоченных фич
        :param theme: тематика группы, string
        """
        self.__id = group_id
        self.__name = name
        self.__features = features
        self.__theme = theme

    def get_id(self):
        return self.__id

    def get_name(self):
        return self.__name

    def get_thema(self):
        return self.__features

    def get_interests(self):
        return self.__theme


class Post:
    """
    Класс, описывающий пост вк
    Класс неизменяемый, все параметры задаются при создании один раз. Для всех параметров написаны геттеры.
    Других функций у класса нет, нужен для более удобного представления группы
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
