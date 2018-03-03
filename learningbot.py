import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB

##Following this tutorial: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

raw_data = 'combined_data.tsv'
df = pd.read_csv(raw_data, encoding='utf-8', sep='\t')
categories = ['human', 'bot']

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df.body)
print(X_train_counts.shape)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, df.label)
docs_new = ['Hello I am not a bot', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))
