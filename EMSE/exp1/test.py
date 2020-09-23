import sys

# test_string = "update doc. fix #330 --- -也可以直接 `./mvnw clean package -DskipTests`打包，生成的zip在 `packaging/target/` 下面。但是注意`as.sh`启动加载的是`~/.arthas/lib`下面的版本。 +也可以直接 `./mvnw clean package -DskipTests`打包，生成的zip在 `packaging/target/` 下面。但是注意`as.sh`启动加载的\x08是`~/.arthas/lib`下面的版本。"
# print(word_tokenize(test_string))
# print(re.split('([^a-zA-Z0-9])',test_string))
# btk = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# print(btk.tokenize(test_string))
#
#
# def cjk_detect(texts):
#     # korean
#     if re.search("[\uac00-\ud7a3]", texts):
#         return "ko"
#     # japanese
#     if re.search("[\u3040-\u30ff]", texts):
#         return "ja"
#     # chinese
#     if re.search("[\u4e00-\u9FFF]", texts):
#         return "zh"
#     return None
sys.path.append("../..")
sys.path.append("../../..")
from EMSE.BERTDataReader import load_examples

examples = load_examples("G:\\Document\\InterMingualTraceGitData\\git_projects\\alibaba\\arthas", model=None)
print(len(examples))
