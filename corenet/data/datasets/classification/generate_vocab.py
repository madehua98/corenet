from typing import Any, List, Mapping, Tuple
from corenet.data.datasets.utils.text import caption_preprocessing
#from corenet.data.datasets.classification.wordnet_tagged_classification import check_valid_noun_synset, extract_pos_offset_info_from_synset
import pickle
import sys
try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus.reader.wordnet import Synset
    from nltk.stem import WordNetLemmatizer
    #nltk.download('averaged_perceptron_tagger')
    NLTK_INSTALLED = True
except ModuleNotFoundError:
    wn = None
    Synset = None
    WordNetLemmatizer = None

    NLTK_INSTALLED = False

vocab_dict = {}


def read_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines

def get_vocab(captions_str: str, vocab_dict) -> List[int]:

    captions_str = caption_preprocessing(captions_str)
    # process caption and find synsets

    tagged_words = nltk.pos_tag(nltk.word_tokenize(captions_str))
    lemmatzr = WordNetLemmatizer()
    labels = []
    for word, pos in tagged_words:
        # use lemmatizer to reduce text ambiguity.
        # words like bicycle and bicycles are converted to bicycle
        try:
            word = lemmatzr.lemmatize(word)
            noun_synset = wn.synset(f"{word}.n.01")
        except Exception as e:
            # No lemma 'is' with part of speech 'n', then nltk.corpus.reader.wordnet.WordNetError is raised.
            # Skip such cases
            continue

        if not check_valid_noun_synset(noun_synset, word):
            continue
        noun_synset = extract_pos_offset_info_from_synset(noun_synset)
        if noun_synset not in vocab_dict.keys():
            vocab_dict[noun_synset] = 1
        else:
            vocab_dict[noun_synset] += 1
    return vocab_dict

# captions_str_list = []
# captions_filaname = '/media/fast_data/Food2k_complete/food2k_label2name_en.txt'
# captions = read_txt(captions_filaname)
# for caption in captions:
#     _, captions_str = caption.strip().split('--')
#     captions_str_list.append(captions_str)

# for captions_str in captions_str_list:
#     vocab_dict = get_vocab(captions_str, vocab_dict)

# vocab_dict_sorted = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
# print(len(vocab_dict_sorted))

# file_path = 'corenet/data/datasets/classification/food2k_vocab.pkl'
# with open(file_path, 'wb') as file:
#     pickle.dump(vocab_dict_sorted, file)

# file_path = 'corenet/data/datasets/classification/food2k_vocab.pkl'

# # 读取pkl文件
# with open(file_path, 'rb') as file:
#     vocab = pickle.load(file)

# # 打印文件内容
# print(vocab)