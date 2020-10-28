import argparse
import sys

import pandas as pd

sys.path.append("..")
sys.path.append(".")
from metrices import metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv","-c", required=True, help="the csv file with s_id, t_id, pred, label" )
    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    m = metrics(df, output_dir=".")

    pk = m.precision_at_K(3)
    best_f1, best_f2, details = m.precision_recall_curve("pr_curve.png")
    map = m.MAP_at_K(3)
    ap = m.AP()
    print(
        "map={}, ap={}, f1={}".format(map, ap, best_f1)
    )
