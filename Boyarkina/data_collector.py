class Group:
    """
    Класс, описывающий группу вк, ее фичи и тематику
    id, name - строки для идентификации группы
    features - собранные фичи
    theme - тематика группы

    Класс неизменяемый, все параметры задаются при создании один раз. Для всех параметров написаны геттеры.
    Других функций у класса нет, нужен для более удобного представления группы
    """

    def __init__(self, group_id, name, features, theme):
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
