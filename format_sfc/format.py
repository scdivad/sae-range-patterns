import json

incl_patch = True
file_path = "../data/codereason/key/data_len5_digit1.json"
output_file_path = "formatted_sva.json"
# output_file_path = "formatted_key.json"
# output_file_path = "formatted_key_no_traceback.json"

def get_label_sva(prefix, answer):
    # if answer in [' go',  ' are', ' do',  ' have']:
    #     return "plural"
    # elif answer in [' goes', ' is', ' does', ' has']:
    #     return "singular"
    if answer in [' go',  ' are', ' do',  ' have']:
        return 0
    elif answer in [' goes', ' is', ' does', ' has']:
        return 1
    return -1

def get_label_is_answer(prefix, answer):
    return answer

def format_example(prefix, answer, get_label):
    label = get_label(prefix, answer)
    if label == -1:
        print("returning none")
        return None
    return {
        'prompt': prefix,
        'label': label,
    }

with open(file_path, "r") as f, open(output_file_path, "w") as out_f:
    get_label = get_label_sva
    for line in f:
        record = json.loads(line)
        if (tmp := format_example(record['clean_prefix'], record['clean_answer'], get_label)):
            out_f.write(json.dumps(tmp) + "\n")
        if incl_patch and (tmp := format_example(record['patch_prefix'], record['patch_answer'], get_label)):
            out_f.write(json.dumps(tmp) + "\n")
# with open(file_path, "r") as f, open(output_file_path, "w") as out_f:
#     get_label = get_label_is_answer
#     tmp = json.load(f)
#     for record in tmp:
#         if (tmp := format_example(record['clean_prefix'], record['clean_answer'], get_label)):
#             if tmp['label'] != "Traceback":
#                 out_f.write(json.dumps(tmp) + "\n")
#         if incl_patch and (tmp := format_example(record['patch_prefix'], record['patch_answer'], get_label)):
#             if tmp['label'] != "Traceback":
#                 out_f.write(json.dumps(tmp) + "\n")