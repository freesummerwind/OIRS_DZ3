import tensorflow as tf

# constants for database
path_to_sqlite_db = 'database/sqlite_data_1.db'
path_to_group_csv = '/Users/ezuryy/Desktop/oirs/hw3/group_info_df_3.csv'
path_to_post_csv = '/Users/ezuryy/Desktop/oirs/hw3/group_posts_df_3.csv'

group_info_columns = ['id', 'description', 'members_count', 'activity', 'status', 'name', 'screen_name',
                      'is_closed', 'links_count', 'contacts_count', 'age_limits', 'wall',
                      'wiki_page', 'main_section', 'site']

post_columns_csv = ['date', 'id', 'owner_id', 'text', 'comments.count', 'likes.count', 'reposts.count',
                    'views.count', 'photo_url', 'audio_count', 'photo_count', 'video_count', 'doc_count']

post_columns_db = [col.replace('.', '_') for col in post_columns_csv]

# credentials for vk api
version = 5.131
access_token = 'token'

# dataset
# rnn_dataset_path = 'database/vk_posts_rnn_dataset_wothout_stopwords_v2.csv'
rnn_dataset_path = 'database/vk_post_tmp.csv'
warm_inputs = ['как поживать диплом?) уже создать папка на рабочий стол?',
         'это быть круто!!!)))) спасибо всем!!! всем, кто',
         'есть вероятность того, что сегодня быть принималово,']
rnn_path = 'models/dynamic_translator_5'
new_rnn_path = 'models/dynamic_translator_5'

umap_path = 'models/umap_reducer_2.pkl'
hdbscan_path = 'models/hdbscan_model_2.pkl'
top_n_words_path = 'models/top_n_words_2.json'
