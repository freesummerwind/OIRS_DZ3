import tensorflow as tf
from sentence_transformers import SentenceTransformer
import hdbscan
import json
import pickle
from collections import Counter


from config import rnn_path, umap_path, hdbscan_path, top_n_words_path
from parser import parse
from rnn import prepare_text

ERROR_TEXT = 'Can not predict, because there are no posts or posts does not contain text :('


umap_reducer = pickle.load(open(umap_path, "rb"))
clustering_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
hdbscan_model = pickle.load(open(hdbscan_path, "rb"))

rnn_model = tf.saved_model.load(rnn_path)


def get_posts(url):
    _, group_posts_df = parse(url)
    print(group_posts_df)
    inputs = []
    for text in group_posts_df.sort_values(by='date', ascending=False).text.values:
        print('Text inside post:', text[:100])
        if text is not None and len(text) > 5:
            inputs.append(text)
        if len(inputs) > 5:
            break
    if len(inputs) == 0:
        print(ERROR_TEXT)
        return group_posts_df[group_posts_df.text != ''][['owner_id', 'text']][-10:], None
    return group_posts_df[group_posts_df.text != ''][['owner_id', 'text']][-10:], inputs


def get_predict_rnn(inputs):
    inputs = [prepare_text(x, 100) for x in inputs]
    result = rnn_model.translate(tf.constant(inputs))
    print('\n'.join([value.numpy().decode() for value in result]))
    return '\n'.join([value.numpy().decode() for value in result])


def get_predict_cluster(inputs):
    print(type(umap_reducer))
    print(type(hdbscan_model))

    embeddings = clustering_model.encode(inputs, show_progress_bar=True)
    umap_embeddings = umap_reducer.transform(embeddings)
    print(umap_embeddings[0], umap_embeddings.shape)
    print('hdbscan_model.prediction_data_.shape[1]', hdbscan_model.prediction_data_.raw_data.shape[1])
    print('raw data', hdbscan_model.prediction_data_.raw_data[0])
    labels, membership_strengths = hdbscan.approximate_predict(hdbscan_model, umap_embeddings)
    counts = Counter(labels)
    label = str(counts.most_common(1)[0][0])
    with open(top_n_words_path, 'r') as j:
        top_n_words = json.loads(j.read())

    if label in top_n_words:
        result = [x for (x, y) in top_n_words[label]]
        return '  '.join(result)
    else:
        return 'Can not clusterize :('

