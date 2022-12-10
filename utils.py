import re
import nltk
import gensim

from nltk.tokenize.treebank import TreebankWordDetokenizer


def depure_data(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    data = re.sub('\S*@\S*\s?', '', data)  # noqa
    data = re.sub('\s+', ' ', data)  # noqa
    data = re.sub("\'", "", data)  # noqa

    return data


def sent_to_words(sentences):
    for sentence in sentences:
        yield (
            gensim.utils.simple_preprocess(str(sentence), deacc=True)
        )


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


def nltk_setup():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/corpus')
        nltk.data.find('tokenizers/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('corpus')
        nltk.download('stopwords')


def remove_stop_words(text: str):
    stopwords = nltk.corpus.stopwords.words("english")
    return ' '.join([word for word in text.split() if word not in stopwords])
