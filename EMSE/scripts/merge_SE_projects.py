import argparse
import os
import logging
import shutil
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def is_allowed_projects(allowed_projects: List, dir_path: str):
    dir_name = os.path.basename(dir_path)
    if dir_name in allowed_projects or len(allowed_projects) == 0:
        content_list = os.listdir(dir_path)
        if 'issue.csv' in content_list and 'commit.csv' in content_list and 'links.csv' in content_list:
            return True
    return False


def merge_projects(project_list, out_dir):
    header_flag = True
    write_mode = 'w'
    for p in project_list:
        p_name = os.path.basename(p)
        issue_df = pd.read_csv(os.path.join(p, "issue.csv"))
        link_df = pd.read_csv(os.path.join(p, "links.csv"))
        commit_df = pd.read_csv(os.path.join(p, "commit.csv"))

        issue_df['issue_id'] = ["{}_{}".format(p_name, x) for x in issue_df['issue_id']]
        link_df['issue_id'] = ["{}_{}".format(p_name, x) for x in link_df['issue_id']]

        issue_df.to_csv(os.path.join(out_dir, "issue.csv"), mode=write_mode, header=header_flag)
        link_df.to_csv(os.path.join(out_dir, "links.csv"), mode=write_mode, header=header_flag)
        commit_df.to_csv(os.path.join(out_dir, "commit.csv"), mode=write_mode, header=header_flag)

        write_mode = 'a'
        header_flag = False


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
    if len(args.allowed_projects) == 0:
        logger.info("Process {} projects".format(len(project_paths)))
    else:
        for p in args.allowed_projects:
            assert p in args.allowed_projects, p
    merge_projects(project_paths, args.out_dir)
