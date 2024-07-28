import json

from process_data import load_jsonl, save_jsonl

# def load_txt(filename):
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#     return lines
# def save_txt(filename, objects):
#     with open(filename, 'w') as file:
#         for object in objects:
#             file.write(str(object))
#             file.write('\n')

# filename = '/ML-A100/team/mm/models/food200/metadata/class_ingredient.txt'
# lines = load_txt(filename)
# train_ingredient = {}
# for line in lines:
#     parts = line.split()
#     image_name = parts[0]
#     numbers = list(map(int, parts[1:]))
#     num_count = len(numbers)
#     # 找到为1的元素位置
#     positions = [i for i, num in enumerate(numbers) if num == 1]
#     train_ingredient[image_name] = positions

# split_filename = '/ML-A100/team/mm/models/food200/metadata/test_finetune_v2.txt'
# def get_split_data_class(filename):
#     lines = load_txt(filename)
#     data_ingredients = []
#     for data in lines:
#         data_ingredient = {}
#         data = data.split()
#         # 进一步分割文件路径和文件名
#         path_parts = data[0].rsplit('/', 1)
#         label = path_parts[0]
#         data_ingredient[data[0]] = train_ingredient[label]
#         data_ingredients.append(data_ingredient)
#     return data_ingredients
# data_ingredients = get_split_data_class(split_filename)
# train_ingredient_file = '/ML-A100/team/mm/models/food200/metadata/test_ingredient.jsonl'
# save_jsonl(train_ingredient_file, data_ingredients)

# ingredient_file = '/ML-A100/team/mm/models/food101/food101/meta_data/ingredient_list.txt'
# ingredient_list = [i for i in range(174)]
# save_txt(ingredient_file, ingredient_list)



