import os
import subprocess

if __name__ == "__main__":
    root = 'G:\Document\EMSE'
    projs = [x for x in os.listdir(root) if x != 'all' and x != 'chinese_only']
    for p in projs:
        p = os.path.join(root, p)
        print("python split_examples.py -p {} -o {}".format(p, p))
        subprocess.run("python split_examples.py -p {} -o {}".format(p, p).split())
