import os
import pickle
import math
import random
import sys

import pandas

base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_dir,"."))
from Preprocessor import Preprocessor


class Dataset:
    """
    One dataset contains multiple artifact pairs that have links between them.
    """

    def __init__(self, gold_link_sets, round_digit_num=4):
        # Index the gold link sets by their id
        self.gold_link_sets = dict()
        for link_set in gold_link_sets:
            self.gold_link_sets[link_set.get_pair_id()] = link_set

        self.round_digit_num = round_digit_num

    def get_impacted_dataSet(self, replace_list, replace_source_percent=1.0, replace_target_percent=0.0):
        """
        Keep the artifacts and links which have tokens been replaced.
        """
        impacted_link_sets = []
        for link_set_id in self.gold_link_sets.keys():
            link_set = self.gold_link_sets[link_set_id]
            impacted_link_set = link_set.gen_impacted_linkSet(replace_list)  # difference here
            impacted_link_sets.append(impacted_link_set)
        return Dataset(impacted_link_sets)

    def get_replaced_dataSet(self, replace_list, replace_source_percent=1.0, replace_target_percent=0.0):
        """
        Replace part of the english tokens with the given replace list. The data size is equal to origin dataset
        :param replace_list:
        :return:
        """
        replaced_link_sets = []
        for link_set_id in self.gold_link_sets.keys():
            link_set = self.gold_link_sets[link_set_id]
            replaced_link_set = link_set.gen_replaced_linkSet(replace_list, replace_source_percent=0.0,
                                                              replace_target_percent=1.0)  # difference here
            replaced_link_sets.append(replaced_link_set)
        return Dataset(replaced_link_sets)

    def evaluate_link_set(self, gold_link_set_id, eval_link_set):
        """
        Evaluate a set of generate links set against one gold link set by giving the link set id
        :param gold_link_set_id:
        :param eval_link_set:
        :return:
        """
        gold_link_set = self.gold_link_sets[gold_link_set_id]

        gen_links_no_score = set([(x[0], x[1]) for x in eval_link_set])
        gold_links = set(gold_link_set.links)
        tp = len(gen_links_no_score & gold_links)
        fp = len(gen_links_no_score - gold_links)
        fn = len(gold_links - gen_links_no_score)
        total_num = len(gold_link_set.artiPair.source_artif) * len(gold_link_set.artiPair.target_artif)
        tn = total_num - len(gen_links_no_score | gold_links)
        if tp == 0:
            precision = 0
            recall = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        return round(precision, self.round_digit_num), \
               round(recall, self.round_digit_num), \
               round(f1, self.round_digit_num)

    def get_docs(self):
        docs = []
        for link_set in self.gold_link_sets:
            docs.extend(self.gold_link_sets[link_set].get_docs())
        return docs

    def write(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        for link_set_name in self.gold_link_sets:
            write_dir = os.path.join(path, link_set_name)
            if not os.path.isdir(write_dir):
                os.mkdir(write_dir)
            link_set: LinkSet = self.gold_link_sets[link_set_name]
            s_arts = pandas.DataFrame()
            s_arts["issue_id"] = link_set.artiPair.source_artif.keys()
            s_arts["issue_content"] = link_set.artiPair.source_artif.values()
            s_arts.to_csv(os.path.join(write_dir, "issues.csv"))
            try:
                t_arts = pandas.DataFrame()
                t_arts["commit_id"] = link_set.artiPair.target_artif.keys()
                t_arts["commit_content"] = link_set.artiPair.target_artif.values()
                t_arts.to_csv(os.path.join(write_dir, "commits.csv"))
            except Exception:
                # in case commit is too large for pandas
                with open(os.path.join(write_dir, "commits.csv"), 'w', encoding="utf8") as fout:
                    fout.write("{},{}\n".format("commit_id", "commit_content"))

                    for i, t_id in enumerate(link_set.artiPair.target_artif.keys()):
                        if i == 0:
                            continue
                        fout.write("{},{}\n".format(t_id, link_set.artiPair.target_artif[t_id]))

            links = link_set.links
            link_df = pandas.DataFrame()
            link_df["issue_id"] = [x[0] for x in links]
            link_df["commit_id"] = [x[1] for x in links]
            link_df.to_csv(os.path.join(write_dir, "links.csv"))

    def save(self, path):
        with open(path, 'wb') as fout:
            pickle.dumps(self.gold_link_sets, fout)

    def load(self, path):
        with open(path) as fin:
            self.gold_link_sets = pickle.load(fin)

    def __str__(self):
        res = []
        for linkset_id in self.gold_link_sets:
            linkset: LinkSet = self.gold_link_sets[linkset_id]
            link_size = len(linkset.links)
            source_size = len(linkset.artiPair.source_artif)
            target_size = len(linkset.artiPair.target_artif)
            stat = "Linkset Id:{} LinkSize:{}, SourceSize:{}, TargetSize{}".format(linkset_id, link_size,
                                                                                   source_size, target_size)
            res.append(stat)
        return "\n".join(res)


class ArtifactPair:
    """
    Each artifactPair contains 2 type of artifacts in the dataset that can have links in between.
    It only holds the artifact contents
    """

    def __init__(self, source_artif, source_name, target_artif, target_name):
        """

        :param source_artif: a dictionary key is the artifact id , value is the artifact content
        :param source_name: the type of source artifact
        :param target_artif: like wise
        :param target_name:  like wise
        """
        self.source_name = source_name
        self.target_name = target_name
        self.source_artif = source_artif
        self.target_artif = target_artif
        self.source_artif_extra_info = dict()
        self.target_artif_extra_info = dict()

    def get_pair_id(self):
        return self.source_name + "-" + self.target_name

    def get_source_size(self):
        return len(self.source_artif)

    def get_target_size(self):
        return len(self.target_artif)


class LinkSet:
    """
    The links between 2 types of artifacts
    """

    def __init__(self, artiPair: ArtifactPair, links):
        self.artiPair = artiPair
        self.links = links
        self.replacement_info = ""
        self.preprocessor = Preprocessor()

    def get_pair_id(self):
        return self.artiPair.get_pair_id()

    def gen_replaced_linkSet(self, replace_dict, replace_source_percent=1.0, replace_target_percent=0.0):
        replaced_source_artifacts_dict = dict(self.artiPair.source_artif)
        replaced_target_artifact_dict = dict(self.artiPair.target_artif)
        if replace_source_percent > 0:
            for arti_id in self.artiPair.source_artif:
                replaced_source_artifacts_dict[arti_id] = self.replace_tokens(self.artiPair.source_artif[arti_id],
                                                                              replace_dict, replace_source_percent)
        if replace_target_percent > 0:
            for arti_id in self.artiPair.target_artif:
                replaced_target_artifact_dict[arti_id] = self.replace_tokens(self.artiPair.target_artif[arti_id],
                                                                             replace_dict, replace_target_percent)
        replaced_arti_pair = ArtifactPair(replaced_source_artifacts_dict, self.artiPair.source_name,
                                          replaced_target_artifact_dict, self.artiPair.target_name)
        return LinkSet(replaced_arti_pair, self.links)

    def gen_impacted_linkSet(self, replace_dict):
        """
        Prune the artifacts and links to keep the artifacts and links contain replacement.
        But the replacement is not applied to the document.

        :param replace_dict: A dictionary of en-zh
        :param replace_source: replace the english tokens in source artifacts
        :param replace_target:  replace the english tokens in target artifacts
        :return:
        """
        impacted_source_artifacts_dict = dict(self.artiPair.source_artif)
        impacted_target_artifacts_dict = dict(self.artiPair.target_artif)

        for arti_id in self.get_impacted_artifacts(self.artiPair.source_artif,
                                                   replace_dict):  # reserved only the impacted artifacts
            impacted_source_artifacts_dict[arti_id] = self.artiPair.source_artif[arti_id]

        for arti_id in self.get_impacted_artifacts(self.artiPair.target_artif, replace_dict):  # Similar here
            impacted_target_artifacts_dict[arti_id] = self.artiPair.target_artif[arti_id]

        impacted_arti_pair = ArtifactPair(impacted_source_artifacts_dict, self.artiPair.source_name,
                                          impacted_target_artifacts_dict, self.artiPair.target_name)
        impacted_links = self.get_impacted_links(impacted_source_artifacts_dict,
                                                 impacted_target_artifacts_dict)  # Reserve the impacted links.Either source or target are impacted
        return LinkSet(impacted_arti_pair, impacted_links)

    def replace_tokens(self, content, replace_dict, replace_probability=1.0):
        tokens = list(self.preprocessor.get_tokens(content))
        replaced_content = []

        for token in tokens:
            if token in replace_dict:
                fo_words = replace_dict[token]
                if isinstance(fo_words, (list,)):
                    fo_word = random.sample(fo_words, 1)[0]
                else:
                    fo_word = fo_words
                if random.randint(0, 100) / 100.0 <= replace_probability:
                    replaced_content.append(fo_word)
                else:
                    replaced_content.append(token)
            else:
                replaced_content.append(token)
        return " ".join(replaced_content)

    def get_impacted_artifacts(self, origin_artifacts, replace_dict):
        """
        Find the artifacts in oring_artifacts which contains token in replace_dict keys
        :param origin_artifacts:
        :param replace_dict:
        :return:
        """
        impacted = set()
        replace_dict = set(replace_dict.keys())
        for artif in origin_artifacts:
            content = origin_artifacts[artif]
            tokens = set(self.preprocessor.get_tokens(content))
            if len(tokens & replace_dict) > 0:
                impacted.add(artif)
        print("{}/{} are impacted".format(len(origin_artifacts), len(impacted)))
        return impacted

    def get_impacted_links(self, impacted_source, impacted_target):
        impacted_artifacts = []
        impacted_artifacts.extend(impacted_source)
        impacted_artifacts.extend(impacted_target)
        impacted_artifacts = set(impacted_artifacts)
        impacted_links = []
        for link in self.links:
            if link[0] in impacted_artifacts or link[1] in impacted_artifacts:
                impacted_links.append(link)
        impacted_link_info = str(
            len(impacted_links)) + " links are impacted by the replacement, total links num=" + str(
            len(self.links))
        impacted_artifact_info = "impacted source size={}, impacted target size = {}".format(len(impacted_source),
                                                                                             len(impacted_target))
        self.replacement_info = impacted_artifact_info + "\n" + impacted_link_info
        print(self.replacement_info)
        return impacted_links

    def get_docs(self):
        docs = []
        for a in self.artiPair.source_artif:
            docs.append(self.artiPair.source_artif[a])
        for a in self.artiPair.target_artif:
            docs.append(self.artiPair.target_artif[a])
        return docs


class MAP_cal:
    def __init__(self, rank, gold, round_digit_num=4, do_sort=True):
        self.round_digit_num = round_digit_num
        self.rank_gold_pairs = []  # keep data for multiple experiments if necessary in future
        if do_sort:  # for performance consideration in case the the rank is and large and sorted already
            rank = sorted(rank, key=lambda k: k[2], reverse=True)
        rank = [(x[0], x[1], round(x[2], 5)) for x in rank]
        self.rank_gold_pairs.append((rank, gold))

    def recall(self, rank, gold, num):
        included = 0
        slice = set(rank[:num + 1])
        for gold_link in gold:
            if gold_link in slice:
                included += 1
        return included / len(gold)

    def precision(self, rank, gold, num):
        hit = 0
        for i in range(0, num + 1):
            link = (rank[i][0], rank[i][1])
            if link in gold:
                hit += 1
        return hit / (num + 1)

    def __get_average_index(self, gold_link, ranks):
        """
        If multiple links share same score with the gold, then the index of the gold link should be averaged
        :return:
        """
        gold_index = 0
        gold_score = 0
        for i, link in enumerate(ranks):
            if (link[0], link[1]) == gold_link:
                gold_index = i
                gold_score = link[2]
                break
        left_index = gold_index
        right_index = gold_index
        while left_index >= 0 and ranks[left_index][2] == gold_score:
            left_index -= 1
        while right_index < len(ranks) and ranks[right_index][2] == gold_score:
            right_index += 1
        return math.floor((left_index + right_index) / 2)

    def average_precision(self, rank, gold):
        sum = 0
        if len(gold) == 0:
            return 0
        for g in gold:
            g_index = self.__get_average_index(g, rank)
            precision = self.precision(rank, gold, g_index)
            sum += precision
        print(sum)
        return round(sum / len(gold), self.round_digit_num)

    def mean_average_precision(self, rank_gold_pairs):
        sum = 0
        for pair in rank_gold_pairs:
            rank = pair[0]
            gold = pair[1]
            average_precision = self.average_precision(rank, gold)
            sum += average_precision
        return round(sum / len(rank_gold_pairs), 3)

    def run(self):
        return self.mean_average_precision(self.rank_gold_pairs)


class Map_from_file:
    """
    Calculate Map from result file
    """

    def __init__(self, gold_file: str, result_file: str):
        self.gold_file = gold_file
        self.result_file = result_file
        self.rank = []
        self.gold = []
        with open(gold_file) as g_fin:
            for i, line in enumerate(g_fin):
                if i == 0:
                    continue
                source, target = line.strip("\n\t\r").split(",")
                source = source.strip("\n\t\r ")
                target = target.strip("\n\t\r ")
                self.gold.append((source, target))
        with open(result_file) as l_fin:
            for i, line in enumerate(l_fin):
                parts = [x.replace("\'", "").strip() for x in line.strip("\n\t\r\"\(\)").split(",")]
                self.rank.append((parts[0], parts[1], float(parts[2].strip())))

    def run(self):
        return MAP_cal(self.rank, self.gold, do_sort=False, round_digit_num=8).run()


if __name__ == "__main__":
    en_zh_res = "G:\Projects\InterLingualTrace\main\\reborn\experiments\Experiment2\\results\\alibaba\canal\\vsm-zh-en\\vsm_issues-commits_link_score.txt"
    link_file = "G:\Projects\InterLingualTrace\main\\reborn\github_project_crawl\git_projects\\alibaba\canal\links.csv"
    translated = "G:\Projects\InterLingualTrace\main\\reborn\experiments\Experiment2\\results\\alibaba\canal\\vsm-translate\\vsm_issues-commits_link_score.txt"
    file_map_en_zh = Map_from_file(link_file, en_zh_res)
    print(file_map_en_zh.run())
    file_map_en_translated = Map_from_file(link_file, translated, )
    print(file_map_en_translated.run())
