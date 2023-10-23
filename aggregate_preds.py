import os
import glob
import json

input_path = "./out/sup"
output_file = "./out/sup_class_labels.json"

data = {}

for fpath in glob.glob("{}/*json".format(input_path)):
    fname = os.path.basename(fpath)
    with open(fpath) as f:
        d = json.load(f)

    k = os.path.splitext(fname)[0]
    print(k)
    data[k] = d

with open(output_file, 'w') as f:
    json.dump(data, f)

