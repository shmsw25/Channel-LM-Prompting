import os
from collections import Counter
# 'anli', 'glue-mnli', 'glue-qnli', 'glue-rte', 'glue-wnli', 'scitail', 'sick', 'superglue-cb'
task = "sick"
data_dir = "data/k-shot/{}/16-13/0/test.tsv"

labels = []
with open(data_dir.format(task), "r") as f:
    for line in f:
        labels.append(line.strip().split("\t")[-1])

counts = Counter(labels)
print("Label distribution for {}".format(task))
print(counts)