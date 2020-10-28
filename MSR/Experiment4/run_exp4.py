import os
import sys

from experiment4 import Experiment4



def run_CLG():
    exp = Experiment4(model_type='gvsm', lang_code='zh', repo_path='chinese_only', use_translated_data=False,
                      term_similarity_type='cl_w')
    exp.run()


def run_WEG():
    exp = Experiment4(model_type='gvsm', lang_code='en', repo_path='chinese_only', use_translated_data=True,
                      term_similarity_type='gensim_wv')
    exp.run()


if __name__ == "__main__":
    run_CLG()
