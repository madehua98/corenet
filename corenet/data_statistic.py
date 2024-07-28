from process_data import save_jsonl, load_jsonl
import tqdm

filename = '/ML-A100/team/mm/models/catlip_data/datacomp_1b/captions.jsonl'
datas = load_jsonl(filename)



def get_text_length(datas):
    total_length = 0
    for idx, data in tqdm(enumrate(datas), total=len(datas)):
        text = data["captions"]
        total_length += length(text)
    average_length = total_length/len(datas)
    return average_length
average_length = get_text_length(datas)
print(average_length)