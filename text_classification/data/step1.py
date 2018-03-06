"""
Extract the datasets.
"""
import re
import gzip
import json
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups

words = set()
dataset = defaultdict(list)

for subset in ["train", "test"]:
    newsgroups = fetch_20newsgroups(subset=subset, categories=["sci.crypt", "sci.electronics", "sci.med", "sci.space"])
    for target, text in zip(newsgroups.target, newsgroups.data):
        target = newsgroups.target_names[target]
        dataset["%sX" % subset].append(text)
        dataset["%sY" % subset].append(target)
        words.update(re.sub(r'[^\w]', ' ', text).lower().split())

with open("newsgroups.json", "wt") as fout:
    json.dump(dataset, fout)

wordvec = {}
with gzip.open("glove.6B.50d.txt.gz", "rt", encoding="utf8") as fin:
    for line in fin:
        word, *vector = line.split()
        if word in words:
            vector = list(map(float, vector))
            wordvec[word] = vector
with open("glove.json", "wt") as fout:
    json.dump(wordvec, fout)
