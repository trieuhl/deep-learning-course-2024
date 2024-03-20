# import Word2Vec loading capabilities
from gensim.models import KeyedVectors

def load_model():

    # Creating the model
    embed_lookup = KeyedVectors.load_word2vec_format('word2vec_model/GoogleNews-vectors-negative300-SLIM.bin',
                                                     binary=True)

    return