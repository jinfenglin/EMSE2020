from gensim import corpora, models, matutils
from gensim.models import TfidfModel

from Preprocessor import Preprocessor
from model import Model


class VSM(Model):
    def __init__(self, fo_lang_code):
        super().__init__(fo_lang_code)
        self.tfidf_model: TfidfModel = None

    def build_model(self, docs):
        print("Building VSM model...")
        docs_tokens = []
        cnt = 0
        for doc in docs:
            # print(cnt, len(docs))
            cnt += 1
            docs_tokens.append(self.preprocessor.get_stemmed_tokens(doc, self.fo_lang_code))
            #docs_tokens.append(self.preprocessor.get_tokens(doc, self.fo_lang_code))
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
        print("Finish building VSM model")

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        doc1_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc1_tk)]
        doc2_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc2_tk)]
        return matutils.cossim(doc1_vec, doc2_vec)

    def get_model_name(self):
        return "VSM"

    def get_word_weights(self):
        dfs = self.tfidf_model.dfs
        idfs = self.tfidf_model.idfs
        res = []
        for termid in dfs:
            word = self.tfidf_model.id2word[termid]
            idf = idfs.get(termid)
            res.append((word, idf))
        return res


if __name__ == "__main__":
    docs = [
        'this is a test',
        'test assure quality',
        'test is important',
    ]
    vsm = VSM("en")
    vsm.build_model(docs)
    preprocessor = Preprocessor()
    new_doc1 = preprocessor.get_stemmed_tokens("software quality rely on test", "en")
    new_doc2 = preprocessor.get_stemmed_tokens("quality is important", "en")
    new_doc3 = preprocessor.get_stemmed_tokens("i have a pretty dog", "en")
