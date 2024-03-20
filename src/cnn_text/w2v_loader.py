from gensim.models import KeyedVectors


class W2VLoader():
    def __init__(self, w2v_path='data/word2vec_model/GoogleNews-vectors-negative300.bin'):
        self.embed_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.pretrained_words = self.get_pretrained_words()
        self.vocab_size = len(self.pretrained_words)
        self.embedding_dim = len(self.embed_model[self.pretrained_words[0]])  # 300-dim vectors

    def get_pretrained_words(self):
        # store pretrained vocab
        pretrained_words = []
        for word in self.embed_model.vocab:
            pretrained_words.append(word)
        return pretrained_words
