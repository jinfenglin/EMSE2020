import os
import re
import sys

sys.path.append("../")
from Datasets import ArtifactPair, LinkSet, Dataset


def limit_artifacts_in_links(dataset: Dataset):
    """
    Remove the artifacts which did not appear in the golden links
    :param dataset:
    :return:
    """
    modified_link_sets = []
    data_set_infos = []
    for linkset_id in dataset.gold_link_sets:
        link_set: LinkSet = dataset.gold_link_sets[linkset_id]
        source_dict: dict = link_set.artiPair.source_artif
        target_dict: dict = link_set.artiPair.target_artif
        links = link_set.links

        gold_artif_set = set()
        for (s, t) in links:
            gold_artif_set.add(s)
            gold_artif_set.add(t)

        limited_source_dict = dict()
        for s_art in source_dict.keys():
            if s_art in gold_artif_set:
                limited_source_dict[s_art] = source_dict[s_art]
        limited_target_dict = dict()
        for t_art in target_dict.keys():
            if t_art in gold_artif_set:
                limited_target_dict[t_art] = target_dict[t_art]
        modified_artif_pair = ArtifactPair(limited_source_dict, link_set.artiPair.source_name, limited_target_dict,
                                           link_set.artiPair.target_name)
        # Keep the extra information
        modified_link_sets.append(LinkSet(modified_artif_pair, links))
        issue_num = len(modified_artif_pair.source_artif)
        commit_num = len(modified_artif_pair.target_artif)
        issue_commit_info = "{} issues and {} commits remains after limiting artifacts to links...".format(
            issue_num, commit_num)
        data_set_infos.append(issue_commit_info)
        # print(issue_commit_info)
        # candidate_num = issue_num * commit_num
        # base_accuracy = 0
        # if candidate_num > 0:
        #     base_accuracy = len(links) / candidate_num
        # # print("Baseline accuracy is {}/{} = {}".format(len(links), candidate_num, base_accuracy))
    return Dataset(modified_link_sets), "\n".join(data_set_infos)


def readData(issue_path, commit_path, link_path, do_filter=True):
    def all_english(content: str) -> bool:
        def get_en(doc):
            pattern = re.compile("[a-zA-Z]+")
            res = pattern.findall(doc)
            return res

        return len(get_en(content)) == len(content.split())

    issues = dict()
    commits = dict()
    issue_close_time_dict = dict()
    commit_time_dict = dict()
    MIN_DOC_SIZE = 15
    filtered_issued = 0
    filtered_commit = 0
    with open(issue_path, encoding='utf8') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            parts = line.strip("\n\t\r").split(",")
            if len(parts) == 3:
                id, content, close_time = parts
            elif len(parts) == 4:
                _, id, content, close_time = parts
            else:
                raise Exception()
            if (len(content.split()) < MIN_DOC_SIZE) and do_filter:
                filtered_issued += 1
                continue
            issues[id] = content
            issue_close_time_dict[id] = close_time

    with open(commit_path, encoding='utf8') as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            parts = line.strip("\n\t\r").split(",")
            if len(parts) == 4:
                id, summary, content, commit_time = parts
            elif len(parts) == 5:
                _, id, summary, content, commit_time = parts
            else:
                raise Exception()
            commit_content = summary + content
            if (len(commit_content.split()) < MIN_DOC_SIZE) and do_filter:
                filtered_commit += 1
                continue
            commits[id] = commit_content
            commit_time_dict[id] = commit_time
    # print("{} commit are filtered minimal lenght {}".format(filtered_commit, MIN_DOC_SIZE, len(commits)))
    artif_pair = ArtifactPair(issues, "issues", commits, "commits")

    links = []
    origin_link_cnt = 0
    with open(link_path) as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            origin_link_cnt = i
            parts = line.split(",")
            issue_id = parts[-2]
            commit_id = parts[-1]
            issue_id = issue_id.strip("\n\t\r")
            commit_id = commit_id.strip("\n\t\r")
            if issue_id not in issues or commit_id not in commits:
                continue
            link = (issue_id, commit_id)
            links.append(link)
    link_set = LinkSet(artif_pair, links)
    return Dataset([link_set])


class Exp4DataReader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def readData(self, use_translated_data=False, do_filter_on_raw=True) -> Dataset:
        """

        :param use_translated_data: use the translated dataset or origin dataset
        :param do_filter_on_raw: whether applying filtering condition on origin dataset. The filtering operation will be
         mirrored to the translated dataset.
        :return:
        """
        issue_path = os.path.join(self.data_dir, "clean_token_issue.csv")
        commit_path = os.path.join(self.data_dir, "clean_token_commit.csv")
        link_path = os.path.join(self.data_dir, "test", "links.csv")
        origin_dataset = readData(issue_path, commit_path, link_path, do_filter=False)
        if use_translated_data:
            issue_path = os.path.join(self.data_dir, "translated_token_issue.csv")
            commit_path = os.path.join(self.data_dir, "translated_token_commit.csv")
            trans_dataset = readData(issue_path, commit_path, link_path, do_filter=False)
            # map the translated gold linkset back to origin datset, any filtering on origin dataset will reflect on trans dataset
            for link_set_id in trans_dataset.gold_link_sets:
                origin_link_set: LinkSet = origin_dataset.gold_link_sets[link_set_id]
                trans_link_set: LinkSet = trans_dataset.gold_link_sets[link_set_id]

                trans_link_set.links = origin_link_set.links
                remove_elements = []
                for s_id in trans_link_set.artiPair.source_artif:
                    if s_id not in origin_link_set.artiPair.source_artif:
                        remove_elements.append(s_id)
                for s_id in remove_elements:
                    del trans_link_set.artiPair.source_artif[s_id]
                remove_elements = []
                for t_id in trans_link_set.artiPair.target_artif:
                    if t_id not in origin_link_set.artiPair.target_artif:
                        remove_elements.append(t_id)
                for t_id in remove_elements:
                    del trans_link_set.artiPair.target_artif[t_id]
            return trans_dataset
        else:
            return origin_dataset

if __name__ == "__main__":
    r = Exp4DataReader("G:\Document\EMSE\chinese_only")
    d = r.readData()
    print(d)