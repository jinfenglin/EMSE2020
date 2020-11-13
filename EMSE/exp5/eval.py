import logging
import os
import sys
import time

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../../")


from EMSE.BERTDataReader import load_examples
from transformers import BertConfig
from EMSE.metrices import metrics
from EMSE.models import TBertSiamese
from EMSE.utils import results_to_df

MODEL_FNAME = "t_bert.pt"
OPTIMIZER_FNAME = "optimizer.pt"
SCHED_FNAME = "scheduler.pt"
ARG_FNAME = "training_args.bin"
logger = logging.getLogger(__name__)


def test(args, model, eval_examples, cache_file, batch_size=1000):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    retr_res_path = os.path.join(args.output_dir, "raw_result.csv")

    if args.overwrite or not os.path.isfile(cache_file):
        chunked_retrivial_examples = eval_examples.get_chunked_retrivial_task_examples(
            chunk_query_num=args.chunk_query_num,
            chunk_size=batch_size)
        torch.save(chunked_retrivial_examples, cache_file)
    else:
        chunked_retrivial_examples = torch.load(cache_file)
    retrival_dataloader = DataLoader(chunked_retrivial_examples, batch_size=args.per_gpu_eval_batch_size)
    res = []
    for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
        nl_ids = batch[0]
        pl_ids = batch[1]
        labels = batch[2]
        nl_embd, pl_embd = eval_examples.id_pair_to_embd_pair(nl_ids, pl_ids)

        with torch.no_grad():
            model.eval()
            nl_embd = nl_embd.to(model.device)
            pl_embd = pl_embd.to(model.device)
            sim_score = model.get_sim_score(text_hidden=nl_embd, code_hidden=pl_embd)
            for n, p, prd, lb in zip(nl_ids.tolist(), pl_ids.tolist(), sim_score, labels.tolist()):
                res.append((n, p, prd, lb))

    df = results_to_df(res)
    df.to_csv(retr_res_path)
    m = metrics(df, output_dir=args.output_dir)
    return m


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../data/code_search_net/python", type=str,
        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--model_path", default=None, help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--test_num", type=int,
                        help="The number of true links used for evaluation. The retrival task is build around the true links")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_dir", default="./evaluation/test", help="directory to store the results")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the cached data")
    parser.add_argument("--exp_name", help="id for this run of experiment")
    parser.add_argument("--lm", help="language model")
    parser.add_argument("--chunk_query_num", default=-1, type=int,
                        help="The number of queries in each chunk of retrivial task")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    return args


if __name__ == "__main__":
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    res_file = os.path.join(args.output_dir, "./raw_res.csv")

    cache_dir = os.path.join(args.data_dir, "cache")
    cached_file = os.path.join(cache_dir, "test_examples_cache.dat".format())

    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model = TBertSiamese(BertConfig(), args.lm)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, MODEL_FNAME)
        model.load_state_dict(torch.load(model_path))

    logger.info("model loaded")
    start_time = time.time()
    test_examples = load_examples(os.path.join(args.data_dir, "test"), model=model)
    test_examples.update_embd(model)
    m = test(args, model, test_examples, "cached_siamese2_test")
    exe_time = time.time() - start_time
    m.write_summary(exe_time)
