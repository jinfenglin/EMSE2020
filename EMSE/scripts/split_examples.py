"""
split dataset into train/valid/test
"""
import argparse
import os

import pandas as pd
from pandas import DataFrame

"""
-p G:\Document\EMSE\all
-o G:\Document\EMSE\all
-l links.csv
-s issue.csv
-t commit.csv
"""


def create_subset(link: DataFrame, source: DataFrame, target: DataFrame, args, sub_set_name):
    source = source[source.iloc[:, args.sid_col].isin(link.iloc[:, 0])]
    target = target[target.iloc[:, args.tid_col].isin(link.iloc[:, 1])]
    subset_dir = os.path.join(args.out_dir, sub_set_name)
    if not os.path.isdir(subset_dir):
        os.makedirs(subset_dir)
    source.to_csv(os.path.join(subset_dir, args.source))
    target.to_csv(os.path.join(subset_dir, args.target))
    link.to_csv(os.path.join(subset_dir, args.link))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", "-p", required=True, help="The project contains Issue, Commits and links")
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory path")
    parser.add_argument("--link", "-l", default="links.csv",
                        help="file name of links, which must have 2 columns with artifact ids")
    parser.add_argument("--source", "-s", default="issue.csv", help="file name for source artifacts")
    parser.add_argument("--target", "-t", default="commit.csv", help="file name for target artifacts")

    parser.add_argument("--sid_col", "-si", default=1, help="index for the id column in source artifact")
    parser.add_argument("--tid_col", "-ti", default=1, help="index for the id column in target artifact")
    args = parser.parse_args()

    link_df = pd.read_csv(os.path.join(args.project_dir, args.link))

    train_index = int(0.8 * len(link_df))
    valid_index = int(0.9 * len(link_df))

    train_links = link_df.iloc[:train_index, :]
    valid_links = link_df.iloc[train_index:valid_index, :]
    test_links = link_df.iloc[valid_index:, :]

    sart_df = pd.read_csv(os.path.join(args.project_dir, args.source))
    tart_df = pd.read_csv(os.path.join(args.project_dir, args.target))
    create_subset(train_links, sart_df, tart_df, args, 'train')
    create_subset(valid_links, sart_df, tart_df, args, 'valid')
    create_subset(test_links, sart_df, tart_df, args, 'test')
