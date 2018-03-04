from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

#https://machinelearningmastery.com/clean-text-machine-learning-python/
###https://stackoverflow.com/questions/8897593/similarity-between-two-text-documents###
stop_words = stopwords.words('english')
twttokenizer = TweetTokenizer(preserve_case=False)
stemmer = nltk.PorterStemmer()


def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]


def normalize(text):
    answer = text.splitlines()
    comment = ' '.join(answer)
    return stem_tokens(twttokenizer.tokenize(comment))


def cosine_sim(list_of_text):
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stop_words)
    tfidf = vectorizer.fit_transform(list_of_text)
    return ((tfidf * tfidf.T).A)[0, 1]
