import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

datasets = ["rsivl", "visc", "objects", "scenes", "art", "int", "sup"]
output = {}

for dataset in datasets:

    with open("./out/{}_class_labels.json".format(dataset)) as f:
        d = json.load(f)

    unique_d = {}

    for k, v in d.items():
        unique_d[k] = set([x['category'] for x in v])

    output[dataset] = pd.DataFrame({
        "filename": [k for k, v in sorted(d.items())],
        "num_classes": [len(v) for k, v in sorted(d.items())],
        "num_unique_classes": [len(v) for k, v in sorted(unique_d.items())]
    })

    # plt.hist(df["num_classes"], label="non-unique")
    # plt.hist(df["num_unique_classes"], label="unique")
    # plt.legend()
    # plt.show()

with open("./out/fcclip_labels.p", "wb") as f:
    pickle.dump(output, f)