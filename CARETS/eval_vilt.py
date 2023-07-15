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

## eval result for one test
curr_test_name = 'negation_consistency'
## epoch represents the total number of epochs; c is a counter for negation in this case
## For running metrics of other fine-tuned models (it means using different data for fine-tuning): change curr_test_name, epochs, path of predictions and conditions for c
## c=0: antonym; c=1: ontological; c=2: phrasal; c=3: symmetry; c=4: negation
epochs = 10
for epoch in range(epochs):
    preds = np.load(f'vilt_res1/{curr_test_name}_{epoch}.npy', allow_pickle=True).item()

    #orig_ques = np.load("Orig_ques.npy", allow_pickle=True).item()
    #per_ques = np.load("Per_ques.npy", allow_pickle=True).item()

    c=0
    for test_name, split in dataset.splits:
        for question in split:
            question_id = question['question_id']
            if question_id in preds.keys():
                continue
            preds[question_id] = 'unknown'


    for test_name, split in dataset.splits:
        ## add wups
        # if test_name == 'antonym_consistency' or test_name=='ontological_consistency' or test_name=='phrasal_invariance':
            # continue
        if c == 4:
            
            wups = split.total_wups(preds)
            accuracy = split.total_accuracy(preds)
            consistency = split.evaluate(preds)
            comprehensive_accuracy = split.comprehensive_accuracy(preds)
            eval_type = split.eval_type

            print(f'{test_name.ljust(24)}: accuracy: {accuracy:.3f}, {eval_type.ljust(24)}:' + \
            f' {consistency:.3f}, comprehensive_accuracy: {comprehensive_accuracy:.3f}, wups: {wups:.3f}')
        c+=1
        