import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc":  "Size"},axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


df = pd.read_csv('database/clean_group_post.csv')
df['clean_texts'] = df['clean_texts'].astype(str)
data = df.clean_texts.values  # [:100]
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)


reducer = umap.UMAP(n_neighbors=15,
                    n_components=5,
                    metric='cosine')
umap_embeddings = reducer.fit_transform(embeddings)

cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',
                          cluster_selection_method='eom',
                          prediction_data=True).fit(umap_embeddings)

pickle.dump(cluster, open("models/hdbscan_model_2.pkl", "wb"))
loaded_cluster = pickle.load(open("models/hdbscan_model_2.pkl", "rb"))
print(type(loaded_cluster))

# Подготовка данных
reducer_1 = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
umap_data = reducer_1.fit_transform(embeddings)
print(umap_data[0], umap_data.shape)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

pickle.dump(reducer, open("models/umap_reducer_2.pkl", "wb"))
loaded_reducer = pickle.load(open("models/umap_reducer_2.pkl", "rb"))

docs_df = pd.DataFrame(data, columns=["Doc"])
docs_df['Topic'] = cluster.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_df['Doc'] = docs_df['Doc'].astype(str)
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})


tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
words = count.get_feature_names_out()
labels = list(docs_per_topic.Topic)


top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

for i in range(20):
    # Calculate cosine similarity
    similarities = cosine_similarity(tf_idf.T)
    np.fill_diagonal(similarities, 0)

    # Extract label to merge into and from where
    topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
    topic_to_merge = topic_sizes.iloc[-1].Topic
    topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

    # Adjust topics
    docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
    docs_df.Topic = docs_df.Topic.map(map_topics)
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    # Calculate new topic words
    m = len(data)
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

with open('models/top_n_words_2.json', 'w') as fp:
    json.dump(top_n_words, fp)

print(top_n_words[-1])
