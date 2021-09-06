import os
import subprocess

if __name__ == "__main__":
    data_source = "/afs/crc.nd.edu/user/j/jlin6/data/EMSE"
    n = 2
    for i in range(n):
        output_dir = f"/afs/crc.nd.edu/user/j/jlin6/data/EMSE_revision/round_{i}"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        projs = [x for x in os.listdir(data_source) if x == 'chinese_only']
        for p in projs:
            proj = os.path.join(data_source, p)
            out = os.path.join(output_dir, p)
            print("python split_examples.py -p {} -o {}".format(p, p))
            subprocess.run(
                f"python split_examples.py -p {proj} -o {out} --sid_col 0 --tid_col 0".split()
            )
