# Применение методов машинного обучение для предсказания темы следующего поста в группе вконтакте

В данной работе: 
- [x] Создан интерфейс для парсинга страниц Вконтакте
- [x] Построена модель кластеризации постов ([статья](https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970))
- [x] Обучена модель seq2seq на основе RNN ([прообраз из документации](https://www.tensorflow.org/text/tutorials/nmt_with_attention?hl=ru))
- [x] Создан интерфейс для предсказания темы поста по ссылке на группу

## Ниже часть для меня
### Версии

```angular2html
dynamic_translator 50k размер словаря, мало, не запомнил слова
dynamic_translator_2 100k размер словаря
dynamic_translator_3 50k размер словаря
dynamic_translator_4 50k размер словаря, убрать короткие строки
```
