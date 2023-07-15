## Author: Minxuan Qin
import random
import numpy as np
import sys
# sys.path.append('/cluster/scratch/minqin/hf_vqa/CARETS/')
from carets import CaretsDataset
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig

from PIL import Image

dataset = CaretsDataset('CARETS/configs/default.yml')
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

## Orig/Per_ques.npy: dict: id->ques_obj
def collect_orig_ques():
    original_questions = dict()
    for test_name, split in dataset.splits:
        for question in split:
            if not question['perturbed']:
                question_id = question['question_id']
                if question_id not in original_questions.keys():
                    original_questions[question_id] = question

    print("Original questions:", len(original_questions))

    np.save('Orig_ques.npy', original_questions)

def collect_perturbed_ques():
    i=0
    per_questions = dict()
    for test_name, split in dataset.splits:
        for question in split:
            i+=1
            if question['perturbed']:
                question_id = question['question_id']
                if question_id not in per_questions.keys():
                    per_questions[question_id] = question

    print("Perturbed questions:", len(per_questions))
    np.save('Per_ques.npy', per_questions)

## parrot_data/orig/per.npy: input for parrot
def generate_parrot_pair():
    output_list = []
    orig = np.load('Orig_ques.npy', allow_pickle=True).item()
    per = np.load('Per_ques.npy', allow_pickle=True).item()
    for test_name, split in dataset.splits:
        ## key:sent, value: ids
        orig_ques_dict = dict()
        ## pairs of orig_ques, key:sent, value: pairs_ids
        ## key:orig_sent, value:all perturbed sents
        per_ques_dict = dict()
        for question in split:
            q_id = question['question_id']
            sent = question['sent']
            p_id = question['paired_question_id']
            if not question['perturbed']:
                if sent not in orig_ques_dict.keys():
                    orig_ques_dict[sent] = [p_id]
                else:
                    orig_ques_dict[sent].append(p_id)
        output_list.append(orig_ques_dict)
    
    np.save('parrot_orig.npy', output_list)
    per_list = []
    for item_dict in output_list:
        c = 0
        per_dict = dict()
        for item_k, item_v in item_dict.items():
            ind = 0
            if len(item_v) > 1:
                c += 1
            for i in item_v:
                if ind == 0:
                    per_dict[item_k] = [per[i]['sent']]
                else:
                    per_dict[item_k].append(per[i]['sent'])
                ind += 1
        per_list.append(per_dict)
        print('duplication:', c)
    np.save('parrot_per.npy', per_list)

## orig/per_dict.npy; orig/per_q.npy
def collect_ques_of_test():
    res_list_orig = []
    res_list_per = []
    for test_name, split in dataset.splits:
        temp_orig_dict = dict()
        temp_per_dict = dict()

        for question in split:
            q_id = question['question_id']

            if question['perturbed']:
                temp_per_dict[q_id] = question
            else:
                temp_orig_dict[q_id] = question
        print(test_name)
        print("Original: ", len(temp_orig_dict))
        print("Is same len? ", (len(temp_orig_dict) == len(temp_per_dict)))
        print('\n')

        res_list_orig.append(temp_orig_dict)
        res_list_per.append(temp_per_dict)

    np.save('orig_dict.npy', res_list_orig)
    np.save('per_dict.npy', res_list_per)

    
preds_o = np.load('pred_res/preds_vilt1.npy', allow_pickle=True).item()
orig_dict = np.load("Orig_ques.npy", allow_pickle=True).item()
per_dict = np.load("Per_ques.npy", allow_pickle=True).item()

ids = np.load('error_length.npy', allow_pickle=True)
neg_id = np.load('neg_vilt.npy', allow_pickle=True)
preds = np.load('vilt_res1/negation_consistency_9.npy', allow_pickle=True).item()

c=0

for i in neg_id:
    pred = preds[i]
    pred_len = len(pred.split())
    if pred_len != 1:
        c+=1
print(c)
## output of this code is 0; 
## can change it to the call of any function above to generate any needed data
'''
error_data = np.load('error_data_vilt3.npy', allow_pickle=True)

test = error_data[10]
test_img = test['img']
rgb_img = test_img.convert('RGB')
rgb_img.save('output.png')
print(test['ques'])

error_data_1 = np.load('error_data_visual_resnet.npy', allow_pickle=True)

print(len(error_data_1))
'''