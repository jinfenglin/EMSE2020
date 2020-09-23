import logging
import os
import pandas as pd

from EMSE.data_structures import Examples

logger = logging.getLogger(__name__)


def short_text(text, max_length=1000):
    tokens = [x for x in text.split() if len(x) > 0]
    return " ".join(tokens[: max_length])


class Issue:
    def __init__(self, issue_id: str, desc: str, comments: str, create_time, close_time):
        self.issue_id = issue_id
        self.desc = short_text("" if pd.isnull(desc) else desc)
        # self.desc = desc
        self.comments = short_text("" if pd.isnull(comments) else comments)
        self.create_time = create_time
        self.close_time = close_time

    def to_dict(self):
        return {
            "issue_id": self.issue_id,
            "issue_desc": self.desc,
            "issue_comments": self.comments,
            "closed_at": self.create_time,
            "created_at": self.close_time
        }

    def __str__(self):
        return str(self.to_dict())


class Commit:
    def __init__(self, commit_id, summary, diffs, files, commit_time, short_version=True):
        self.commit_id = commit_id
        self.summary = short_text(summary)
        self.diffs = short_text(diffs)
        self.files = files
        self.commit_time = commit_time

    def to_dict(self):
        return {
            "commit_id": self.commit_id,
            "summary": self.summary,
            "diff": self.diffs,
            "files": self.files,
            "commit_time": self.commit_time
        }

    def __str__(self):
        return str(self.to_dict())


def __read_artifacts(file_path, type):
    df = pd.read_csv(file_path)
    df = df.replace(pd.np.nan, regex=True)
    arti = []
    for index, row in df.iterrows():
        if type == 'commit':
            art = Commit(commit_id=row['commit_id'], summary=row['commit_summary'], diffs=row[' commit_diff'],
                         files=None,
                         commit_time=row['commit_time'])
        elif type == "issue":
            art = Issue(issue_id=row['issue_id'], desc=row['issue_content'], comments=None,
                        create_time=None, close_time=row['closed_at'])
        elif type == "link":
            iss_id = row["issue_id"]
            cm_id = row["commit_id"]
            art = (iss_id, cm_id)
        else:
            raise Exception("wrong artifact type")
        arti.append(art)
    return arti


def read_OSS_examples(data_dir):
    commit_file = os.path.join(data_dir, "commit.csv")
    issue_file = os.path.join(data_dir, "issue.csv")
    link_file = os.path.join(data_dir, "links.csv")
    examples = []
    issues = __read_artifacts(issue_file, "issue")
    commits = __read_artifacts(commit_file, "commit")
    links = __read_artifacts(link_file, "link")
    issue_index = {x.issue_id: x for x in issues}
    commit_index = {x.commit_id: x for x in commits}
    for lk in links:
        if lk[0] not in issue_index or lk[1] not in commit_index:
            continue
        iss = issue_index[lk[0]]
        cm = commit_index[lk[1]]
        iss_text = iss.desc + " " + iss.comments
        cm_text = cm.summary + " " + cm.diffs
        example = {
            "NL": iss_text,
            "PL": cm_text
        }
        examples.append(example)
    return examples


def load_examples(data_dir, model, num_limit=None):
    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    logger.info("Creating examples from dataset file at {}".format(data_dir))
    raw_examples = read_OSS_examples(data_dir)

    if num_limit:
        raw_examples = raw_examples[:num_limit]
    examples = Examples(raw_examples)
    if model:
        examples.update_features(model)
    return examples
