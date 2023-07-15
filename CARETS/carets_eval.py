## Author: Minxuan Qin
import random
import numpy as np
import sys
# sys.path.append('/cluster/scratch/minqin/hf_vqa/CARETS/')
from carets import CaretsDataset
from transformers import ViltProcessor, ViltForQuestionAnswering

dataset = CaretsDataset('CARETS/configs/default.yml')

## load prediction data here
predictions = np.load('preds_blip.npy', allow_pickle=True)
predictions = predictions.item()
print(len(predictions))


for test_name, split in dataset.splits:
    for question in split:
        question_id = question['question_id']
        if question_id in predictions.keys():
            continue
        predictions[question_id] = 'unknown'


for test_name, split in dataset.splits:
    ## add wups
    wups = split.total_wups(predictions)
    accuracy = split.total_accuracy(predictions)
    consistency = split.evaluate(predictions)
    comprehensive_accuracy = split.comprehensive_accuracy(predictions)
    eval_type = split.eval_type
    print(f'{test_name.ljust(24)}: accuracy: {accuracy:.3f}, {eval_type.ljust(24)}:' + \
          f' {consistency:.3f}, comprehensive_accuracy: {comprehensive_accuracy:.3f}, wups: {wups:.3f}')

## Compute orig and perturbed acc & wups
for test_name, split in dataset.splits:
    ## add wups
    orig_wups = split.orig_wups(predictions)
    orig_accuracy = split.orig_accuracy(predictions)
    perturbed_wups = split.perturbed_wups(predictions)
    perturbed_acc = split.perturbed_accuracy(predictions)
    eval_type = split.eval_type
    print(f'{test_name.ljust(24)}: accuracy: {orig_accuracy:.3f}, wups: {orig_wups:.3f}, p_acc: {perturbed_acc:.3f}, p_wups: {perturbed_wups:.3f}')