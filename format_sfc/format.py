import json

formatted_data = []
incl_patch = True
file_path = "../data/sva/rc_train.json"

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

# Specify the output file path
output_file_path = "formatted_rc_train.json"

with open(file_path, "r") as f, open(output_file_path, "w") as out_f:
    get_label = get_label_sva
    for line in f:
        record = json.loads(line)
        if (tmp := format_example(record['clean_prefix'], record['clean_answer'], get_label)):
            formatted_data.append(tmp)
        if incl_patch and (tmp := format_example(record['patch_prefix'], record['patch_answer'], get_label)):
            formatted_data.append(tmp)
        
        # Write the formatted data to the output file as JSON
        out_f.write(json.dumps(tmp) + "\n")
