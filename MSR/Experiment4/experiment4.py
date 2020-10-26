import argparse
import os

base_dir = os.path.dirname(os.path.realpath(__file__))

from models.DataReader import Exp4DataReader, limit_artifacts_in_links
from models.Datasets import MAP_cal, Dataset
from models.GVSM import GVSM
from models.LDA import LDA
from models.LSI import LSI
from models.VSM import VSM


class Experiment4:
    def __init__(self, data_dir, model_type, use_translated_data, repo_path, term_similarity_type, lang_code,
                 link_threshold_interval=5,
                 output_sub_dir=""):

        self.use_translated_data = use_translated_data
        self.data_dir = data_dir
        self.model_type = model_type
        self.repo_path = repo_path
        self.lang_code = lang_code
        self.link_threshold_interval = link_threshold_interval
        self.term_similarity_type = term_similarity_type
        self.output_sub_dir = output_sub_dir

    def get_model(self, model_type, fo_lang_code, docs):
        model = None
        if model_type == "vsm":
            model = VSM(fo_lang_code=fo_lang_code)
            model.build_model(docs)
        elif model_type == "lda":
            model = LDA(fo_lang_code=fo_lang_code)
            model.build_model(docs, num_topics=60, passes=100)
        elif model_type == "gvsm":
            model = GVSM(fo_lang_code=fo_lang_code, term_similarity_type=self.term_similarity_type)
            model.build_model(docs)
        elif model_type == "lsi":
            model = LSI(fo_lang_code=fo_lang_code)
            model.build_model(docs, num_topics=60)
        return model

    def run_model(self, model, dataset: Dataset):
        results = dict()
        for link_set_id in dataset.gold_link_sets:
            link_set = dataset.gold_link_sets[link_set_id]
            source_aritf = link_set.artiPair.source_artif
            target_artif = link_set.artiPair.target_artif
            gen_links = self.get_links(model, source_aritf, target_artif)
            results[link_set_id] = gen_links
        return results

    def get_links(self, trace_model, source_artifact, target_artifact):
        return trace_model.get_link_scores(source_artifact, target_artifact)

    def run(self):
        reader = Exp4DataReader(os.path.join(self.data_dir, self.repo_path))

        dataset = reader.readData(use_translated_data=self.use_translated_data)
        dataset, dataset_info = limit_artifacts_in_links(dataset)
        print(dataset_info)
        model = self.get_model(self.model_type, self.lang_code, dataset.get_docs())
        results = self.run_model(model, dataset)
        for link_set_id in dataset.gold_link_sets:
            print("Processing link set {}".format(link_set_id))
            result = sorted(results[link_set_id], key=lambda k: k[2], reverse=True)
            map = MAP_cal(result, dataset.gold_link_sets[link_set_id].links, do_sort=False).run()
            threshold = 0
            scores = []
            while threshold <= 100:
                filter_links_above_threshold = [x for x in result if x[2] >= threshold / 100]
                eval_score = dataset.evaluate_link_set(link_set_id, filter_links_above_threshold)
                scores.append(eval_score)
                threshold += self.link_threshold_interval

            if not self.use_translated_data:
                trans_postfix = "origin"
            else:
                trans_postfix = "trans"
            write_dir = os.path.join("results", self.output_sub_dir, self.repo_path,
                                     "_".join([self.model_type, trans_postfix]))

            file_name = "{}_{}.txt".format(self.model_type, link_set_id)
            link_score_file = "{}_{}_link_score.txt".format(self.model_type, link_set_id)
            if not os.path.isdir(write_dir):
                os.makedirs(write_dir)
            output_file_path = os.path.join(write_dir, file_name)
            link_score_path = os.path.join(write_dir, link_score_file)

            print("origin MAP=", map)
            print("Origin P,C,F")
            print(scores)

            with open(output_file_path, 'w', encoding='utf8') as fout:
                fout.write(dataset_info + "\n")
                self.write_result(fout, scores, map)
            with open(link_score_path, 'w', encoding='utf8') as fout:
                for link in result:
                    fout.write("{}\n".format(str(link)))

    def write_result(self, writer, prf, map_score):
        writer.write("MAP={}\n".format(map_score))
        writer.write(" P,R,F\n")
        writer.write(str(prf) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment 4 - Merged Chinese dataset")
    parser.add_argument("--data_dir", help="dir for storing the EMSE data", default="G:\Document\EMSE")
    parser.add_argument("--model", help="Model used for experiment")
    parser.add_argument("--repo_path", help="subdir contains data under the data_dir")
    parser.add_argument("--term_similarity_type", help="cl_w or gensim_wv")
    parser.add_argument("--lang_code", help="en,zh etc")
    parser.add_argument("--use_translated_data", action='store_true')
    args = parser.parse_args()
    exp4 = Experiment4(data_dir=args.data_dir, model_type=args.model, use_translated_data=args.use_translated_data,
                       lang_code=args.lang_code,
                       repo_path=args.repo_path, term_similarity_type=args.term_similarity_type)
    exp4.run()
