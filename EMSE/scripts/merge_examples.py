import argparse
import logging
import os
import pandas as pd
import sys

sys.path.append("..")
sys.path.append(".")

"""
For exp2 usage.
Merge the train/valid/test examples instead of the whole projects
"""
from merge_SE_projects import is_allowed_projects

logger = logging.getLogger(__name__)


def collect(in_dir, p_name):
    cm = os.path.join(in_dir, "commit.csv")
    iss = os.path.join(in_dir, "issue.csv")
    lk = os.path.join(in_dir, "links.csv")
    cm_dff, iss_df, lk_df = pd.read_csv(cm), pd.read_csv(iss), pd.read_csv(lk)
    iss_df['issue_id'] = ["{}_{}".format(p_name, x) for x in iss_df['issue_id']]
    lk_df['issue_id'] = ["{}_{}".format(p_name, x) for x in lk_df['issue_id']]
    return cm_dff, iss_df, lk_df


def merge_example(project_list, out_dir):
    types = ['train', 'valid', 'test']
    for t in types:
        cm_dfs, iss_dfs, lk_dfs = [], [], []
        for p in project_list:
            p_name = os.path.basename(p)
            in_dir = os.path.join(p, t)
            cm_df, iss_df, lk_df = collect(in_dir, p_name)
            cm_dfs.append(cm_df)
            iss_dfs.append(iss_df)
            lk_dfs.append(lk_df)
        out_path = os.path.join(out_dir, t)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        pd.concat(cm_dfs, ignore_index=True).to_csv(os.path.join(out_path, "commit.csv"), index=False)
        pd.concat(iss_dfs, ignore_index=True).to_csv(os.path.join(out_path, "issue.csv"), index=False)
        pd.concat(lk_dfs, ignore_index=True).to_csv(os.path.join(out_path, "links.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Root directory for SE projects")
    parser.add_argument("--out_dir", required=True, help="Output directory path")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite the files in output directory if already exist")
    parser.add_argument("--allowed_projects", nargs='+', help="allowed projects to be merged. "
                                                              "The path should be relative path to the root_dir."
                                                              "If it is empty all projects under the root will be merged")

    args = parser.parse_args()
    assert os.path.isdir(args.root_dir)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    elif args.overwrite:
        logger.info("overwriting result in directory {}".format(args.out_dir))
    else:
        raise Exception("Output directory already exist and --overwrite option is set to False")

    project_paths = []
    for root, dirs, files in os.walk(args.root_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not is_allowed_projects(args.allowed_projects, dir_path):
                continue
            project_paths.append(dir_path)
            merge_example(project_paths, args.out_dir)
