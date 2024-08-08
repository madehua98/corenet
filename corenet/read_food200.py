import json

with open('/ML-A100/team/mm/models/food200/recognition/test_IngreLabel.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        for k, v in data.items():
            if v != []:
                print(v)