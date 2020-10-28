import gensim

from gensim import corpora, matutils

from model import Model


class LSI(Model):
    def __init__(self, fo_lang_code):
        super().__init__(fo_lang_code)
        self.lsi = None

    def train(self, docs, num_topics=40):
        docs_tokens = []
        cnt = 0
        for doc in docs:
            # print(cnt, len(docs))
            cnt += 1
            docs_tokens.append(self.preprocessor.get_stemmed_tokens(doc, self.fo_lang_code))
            #docs_tokens.append(self.preprocessor.get_tokens(doc, self.fo_lang_code))
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.lsi = gensim.models.LsiModel(corpus, num_topics= num_topics,id2word=dictionary)

    def build_model(self, docs, num_topics=40):
        self.train(docs, num_topics)

    def get_topic_distrb(self, doc):
        bow_doc = self.lsi.id2word.doc2bow(doc)
        return self.lsi[bow_doc]


    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        dis1 = self.get_topic_distrb(doc1_tk)
        dis2 = self.get_topic_distrb(doc2_tk)
        # return 1 - matutils.hellinger(dis1, dis2)
        return matutils.cossim(dis1, dis2)

    def get_model_name(self):
        return "LSI"


if __name__ == "__main__":
    docs = [
        'this is a test',
        'test assure quality',
        'test is important in software development',
        'quality of service'
    ]
    lsi = LSI(fo_lang_code="en")
    new_doc1 = ["software", 'quality', 'rely', 'test', 'important']
    new_doc2 = ["quality", "is", "important"]
    new_doc3 = ["i", "have", "a", "pretty", "dog"]
    lsi.train(docs)
    dis1 = lsi.get_topic_distrb(new_doc1)
    dis2 = lsi.get_topic_distrb(new_doc2)
    dis3 = lsi.get_topic_distrb(new_doc3)
    print(dis1)
    print(dis2)
    print(dis3)
