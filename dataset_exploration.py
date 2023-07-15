from PIL import Image
import torch
from datasets import load_dataset, get_dataset_split_names
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

## other dataset:
## HuggingFaceM4/VQAv2
## Graphcore/gqa
## flax-community/conceptual-12m-mbart-50-multilingual
'''
train_dataset = load_dataset("HUggingFaceM4/VQAv2", split="train", cache_dir="cache", streaming=False)

sample = train_dataset[2]
sample_q = sample['question']
print(sample_q)

## nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sample_token = nltk.word_tokenize(sample_q)
sample_tags = nltk.pos_tag(sample_token)
print(sample_tags)
'''
'''
df_question = pd.Series(train_dataset['question_type'])
df_language = pd.Series(train_dataset['lang_id'])
df_q = df_question.value_counts().rename_axis('question_type').reset_index(name='counts')
df_l = df_language.value_counts().rename_axis('language').reset_index(name='counts')

## for question type: 65 types, only count top 10
temp_q = df_q.sort_values(by="counts", ascending=False)
# breakpoint()
not_top10 = len(temp_q) - 10
not_top10_sum = temp_q['counts'].tail(not_top10).sum()
q_top10 = temp_q.head(10)

## example: q_top10.plot.pie(...)
#q_top10.plot.pie()
#plt.savefig('ling_vqa_type.png')
df_q.to_csv('question.csv')
'''

orig = np.load('CARETS/orig_dict.npy', allow_pickle=True)
per = np.load('CARETS/per_dict.npy', allow_pickle=True)

## collect and save only questions
## target folder: stats

## pandas dataframe: cols: sent, test_type, perturbed, img_id
import pandas as pd 
## initialize Dataframe
test_type = ['antonym', 'ontological', 'phrasal', 'symmetry', 'negation']

def create_ques_df():
    df = pd.DataFrame(columns=['sent', 'test_type', 'perturbed', 'img_id'])
    for test, item in zip(test_type, orig):
        for ques in item.values():
            sent = ques['sent']
            is_per = ques['perturbed']
            img = ques['img_id']
            df.loc[len(df.index)] = [sent, test, is_per, img]

    for test, item in zip(test_type, per):
        for ques in item.values():
            sent = ques['sent']
            is_per = ques['perturbed']
            img = ques['img_id']
            df.loc[len(df.index)] = [sent, test, is_per, img]

    df.to_json('stats/carets.json')

df = pd.read_json('stats/carets.json', orient='columns', encoding = 'utf-8-sig')
## remaining part see dataset_exploration.ipynb