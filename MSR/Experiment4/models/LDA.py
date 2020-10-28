import gensim
import numpy
from gensim import corpora, matutils

from model import Model


class LDA(Model):
    def __init__(self, fo_lang_code):
        super().__init__(fo_lang_code)
        self.ldamodel = None

    def train(self, docs, num_topics=40, passes=100):
        docs_tokens = []
        cnt = 0
        for doc in docs:
            # print(cnt, len(docs))
            cnt += 1
            docs_tokens.append(self.preprocessor.get_stemmed_tokens(doc, self.fo_lang_code))
            #docs_tokens.append(self.preprocessor.get_tokens(doc, self.fo_lang_code))
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
                                                        passes=passes, alpha='auto', random_state=numpy.random.RandomState(1))

    def build_model(self, docs, num_topics=40, passes=200):
        self.train(docs, num_topics, passes)

    def get_topic_distrb(self, doc):
        bow_doc = self.ldamodel.id2word.doc2bow(doc)
        return self.ldamodel.get_document_topics(bow_doc)

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        """
        :param doc1_tk: Preprocessed documents as tokens
        :param doc2_tk: Preprocessed documents as tokens
        :return:
        """
        dis1 = self.get_topic_distrb(doc1_tk)
        dis2 = self.get_topic_distrb(doc2_tk)
        # return 1 - matutils.hellinger(dis1, dis2)
        return matutils.cossim(dis1, dis2)

    def get_model_name(self):
        return "LDA"


if __name__ == "__main__":
    docs = [
        'this is a test',
        'test assure quality',
        'test is important',

    ]
    lda = LDA(fo_lang_code="en")
    new_doc1 = ["software", 'quality', 'rely', 'test']
    new_doc2 = ["quality", "is", "important"]
    new_doc3 = ["i", "have", "a", "pretty", "dog"]
    lda.train(docs)
    dis1 = lda.get_topic_distrb(new_doc1)
    dis2 = lda.get_topic_distrb(new_doc2)
    dis3 = lda.get_topic_distrb(new_doc3)
    print(dis1)
    print(dis2)
    print(dis3)
    print(matutils.hellinger(dis1, dis2))
    print(matutils.hellinger(dis1, dis3))
