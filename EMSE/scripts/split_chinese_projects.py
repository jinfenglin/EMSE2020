import os
import subprocess

if __name__ == "__main__":
    root = '/afs/crc.nd.edu/user/j/jlin6/data/EMSE'
    projs = [x for x in os.listdir(root) if x != 'all' and x != 'chinese_only']
    for p in projs:
        p = os.path.join(root, p)
        print("python split_examples.py -p {} -o {}".format(p, p))
        subprocess.run("python split_examples.py -p {} -o {} --sid_col 0 --tid_col 0".format(p, p).split())
