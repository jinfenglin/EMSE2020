import re

import many_stop_words
import nltk


class Preprocessor():
    def __init__(self):
        self.zh_pattern = re.compile("[\u4e00-\u9fff]+")
        self.en_pattern = re.compile("[a-zA-Z]+")
        self.fr_pattern = re.compile("[a-zA-ZÀ-ÿ]")
        self.parser = nltk.CoreNLPParser()
        self.java_keywords = set([w.strip('*') for w in """
        abstract    continue    for     new     switch
        assert   default     goto*   package     synchronized
        boolean     do  if  private     this
        break   double  implements  protected   throw
        byte    else    import  public  throws
        case    enum****    instanceof  return  transient
        catch   extends     int     short   try
        char    final   interface   static  void
        class   finally     long    strictfp**  volatile
        const*  float   native  super   while
        """.split()])
        self.customized_stop_words = set(
            "druid version alibaba add update pull copyright "
            "select where when where count over branch "
            "alter todo merge request johnhuang cn master fixed bug canal"
            "马云 什么 现在 急着 直接 指点 可能 时候 "
            "如果 这边 怎么 那些 这些 有助于 好像 没啥 干脆 "
            "因此 可以 是否 哪里 可否 最近 貌似 很多 出现 这个"
            " 但是 还没 这时候 出现 大佬 解答 一下 假设 并且 "
            "一直 但是 郁闷 希望 尽快 没有 由于 学生 来说 有点 "
            "哈哈哈 温绍 周五 下午 写道 一个啥 问题 求助 信息么"
            " 想法 感谢 自己 判断 为什么 义乌 是的 怎么了"
            " 为什么 之后 就是 别人 看到 的话 谢谢 关于 今天"
            " 发现 问题 请问 一旦 这时 能否 大家 快呀 请教 如何"
            " 期待 支持 变得 诡异 起来 使用 导致 比如 一般 需要  "
            "原来 可是 并不 这样 尊敬 大师 您好 学习者 目前 如何"
            " 避免 合并 作者 在一起 这样 一些 其他 很多 绝大部分"
            " 来源于 下面 给做 里面 估计 玩意 误导 我了 不应该 奇怪"
            " 所有 真是 秒回啊 然后 无可 理喻 想问 什么样 另外 帮忙"
            " 看下 看看 以便 然而 求教 各位 如下 以前 为啥 这么 "
            "有影响 林夕 还是 时不时 于是 仍然 十分 想要 以上 不知道 "
            "不需要 上午 不了 孙健 个人 认为 对于 比如说 然后 之前 "
            "因为 后来 温少 此时 感觉 名字 好酷 中文 翻译 德鲁伊 "
            "听到 感觉 里面 英雄 德鲁伊 哈哈 不过 应该 因为 炉石 传说 "
            "里面 难道 因为 项目 创始人 非常 喜欢 卡牌 职业 表示 非常 "
            "好奇 得到 官方 回复 当时 老板 喜欢 名字 好吧 因为 开源 "
            "项目 如此 非常 抽出 宝贵 时间 回答 含量 或者是 像是 一些 怎么办 非常 已经 而非 不然".split())

    def get_stemmer(self, lang_code):
        "danish dutch english finnish french german hungarian italian norwegian porter portuguese romanian russian spanish swedish"
        if lang_code == "en":
            return nltk.SnowballStemmer("english")
        elif lang_code == "fr":
            return nltk.SnowballStemmer("french")
        elif lang_code == "ge":
            return nltk.SnowballStemmer("german")
        elif lang_code == 'it':
            return nltk.SnowballStemmer("italian")
        else:
            return nltk.SnowballStemmer("english")

    def get_zh(self, doc):
        res = self.zh_pattern.findall(doc)
        return res

    def get_en(self, doc):
        res = self.en_pattern.findall(doc)
        return res

    def __clean_doc(self, doc_str):
        doc_str = re.sub("(\d+|[^\w]+)", " ", doc_str, flags=re.UNICODE)
        return doc_str

    def remove_java_keyword(self, tokens):
        return [x for x in tokens if x not in self.java_keywords]

    def get_tokens(self, doc, language="en"):
        def limit_token_min_length(tokens, zh_min=2, en_min=3):
            res = []
            for token in tokens:
                if self.en_pattern.match(token) and len(token) >= en_min:
                    res.append(token)
                elif self.zh_pattern.match(token) and len(token) >= zh_min:
                    res.append(token)
            return res

        tokens = []
        doc = self.__clean_doc(doc)
        res = []
        # if language == "zh":  # maybe a mixture of en and zh
        #     partition_size = 90000
        #     # if doc is too long split it up
        #     for i in range(math.ceil(len(doc) / partition_size)):
        #         try:
        #             doc_parts = doc[i * partition_size: (i + 1) * partition_size]
        #             res = self.parser.tokenize(doc_parts)
        #             res = [x for x in res]
        #         except Exception as e:
        #             print(e)
        #             print("exception when process {}".format(doc_parts))
        # elif language == "ja":
        #     pass
        # elif language == "ko":
        #     self.Kkma
        # else:
        #     res = nltk.word_tokenize(doc)
        res = nltk.word_tokenize(doc)
        for wd in res:
            tokens.extend(self.split_camal_case(wd))
        tokens = [x.lower() for x in tokens]
        tokens = self.remove_java_keyword(tokens)
        tokens = self.remove_stop_word(tokens, language=language)
        tokens = self.remove_stop_word(tokens, language="en")
        tokens = self.remove_stop_word(tokens, stop_words=self.customized_stop_words)
        # tokens = limit_token_min_length(tokens)
        return tokens

    def get_stemmed_tokens(self, doc_str, language="en"):
        en_stemmer = self.get_stemmer("en")
        fo_stemmer = self.get_stemmer(language)

        tokens = self.get_tokens(doc_str, language)
        tokens = [en_stemmer.stem(x) for x in tokens]
        tokens = [fo_stemmer.stem(x) for x in tokens]
        return tokens

    def remove_stop_word(self, token_list, language="en", stop_words=None):
        if stop_words == None:
            if language == "ko":
                language = "kr"
            stop_words = many_stop_words.get_stop_words(language)
        return [x for x in token_list if x not in stop_words]

    def split_camal_case(self, phrase):
        """
        Should not contain whitespace
        :param phrase: phrase in camalcase
        :return:
        """
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', phrase)
        return [m.group(0) for m in matches]


if __name__ == "__main__":
    test_str = "this is a sentence contains CamalCase word HTTPRequest"
    test_str2 = "HTTPRequest"
    test_str3 = "this is sentece for 中英文混杂的句子，不知道可不可以合理地split出来"
    pre = Preprocessor()
    print(pre.get_tokens(test_str))
    print(pre.split_camal_case(test_str2))
    print(pre.get_tokens(test_str3, language="zh"))
