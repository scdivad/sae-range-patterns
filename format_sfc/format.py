import json

num_entries = 10
data = []
with open("../data/sva/rc_train.json", "r") as f:
    i = 0
    for line in f:
        record = json.loads(line)
        data.append(record)
        print(record)
        if i == num_entries:
            break
        i += 1