# "Then, Kristen and Robert went to the restaurant. Kristen gave a computer to"

import json
import random

formatted_data = []
incl_patch = False
file_path = "../data/ioi/ioi_train_21.json"

def swap_names(prompt):
    parts = prompt.split()
    name1 = parts[1]
    name2 = parts[3]
    assert parts[-5] == name1
    parts[1], parts[3] = parts[3], parts[1]
    return " ".join(parts)


# Specify the output file path
output_file_path = "formatted_ioi_train.json"

with open(file_path, "r") as f, open(output_file_path, "w") as out_f:
    num_entries = 6000
    i = 0
    data = json.load(f)
    for record in data:
        prompt = record['clean_prefix']
        label = record['clean_answer']
        # swap = random.random() < 0.5
        tmp1 = {}
        tmp2 = {}
        tmp1['prompt'] = swap_names(prompt)
        tmp1['label'] = 1
        tmp2['prompt'] = prompt
        tmp2['label'] = 0
        out_f.write(json.dumps(tmp1) + "\n")
        out_f.write(json.dumps(tmp2) + "\n")
        i += 1
        if i == num_entries:
            break


'''
    elif task == "ioi":
        batch_size = 16
        max_samples_train = 6000
        max_samples_test = 1000
        acts_train_path = f'acts_format_ioi_train.pt'
        acts_test_path = f'acts_format_ioi_test.pt'
        file_path = "format_sfc/formatted_ioi_train.json"
        example_length = 16
        topk_rules = 2

'''