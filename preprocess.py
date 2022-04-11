import numpy as np
from transformers import BertTokenizerFast
from tqdm import tqdm
import pickle
tokenizer = BertTokenizerFast("vocab/vocab.txt",sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
sep_id = tokenizer.sep_token_id
cls_id = tokenizer.cls_token_id

with open("data/train.txt","r",encoding="utf-8") as f:
    data = f.read()

# 为了区分Linu系统和win系统
if "\r\n" in data:
    train_data = data.split("\r\n\r\n")
else:
    train_data = data.split("\n\n")

data_len = []
data_list=[]


for index, dialogue in enumerate(tqdm(train_data)):
    if "\r\n" in data:
        utterances = dialogue.split("\r\n")
    else:
        utterances = dialogue.split("\n")
    input_ids = [cls_id]  # cls 开头
    for utterance in utterances:
        input_ids+= tokenizer.encode(utterance,add_special_tokens=False)
        input_ids.append(sep_id)
    data_len.append(len(input_ids))
    data_list.append(input_ids)

len_mean =np.mean(data_len)
len_median =np.median(data_len)
len_max =np.max(data_len)
with open("data/train.pkl", "wb") as f:
    pickle.dump(data_list, f)
