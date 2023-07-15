## Author: Minxuan Qin
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from datasets import load_dataset, get_dataset_split_names
import numpy as np

from PIL import Image
import requests
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import AutoProcessor, AutoModelForCausalLM
from nltk.corpus import wordnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("CARETS/vilt_neg_model")

#processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")
#model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")

train_dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", cache_dir="cache", streaming=False)
valid_dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation", cache_dir="cache", streaming=False)


def tokenize_function(examples):
    sample = {}
    sample['inputs'] = processor(images=examples['image'], text=examples['question'], return_tensors="pt")
    sample['outputs'] = examples['multiple_choice_answer']
    return sample

#tokenized_datasets = valid_dataset.map(tokenize_function, batched=True)

#eval_dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=16, shuffle=True)


def inference(saving_path, processor, model, ds):
    sample = ds[2500]
    print(sample)
    image = sample['image']
    question = sample['question']
    answer = sample['multiple_choice_answer']
    ## For manually question:
    #question = 'How many people are there?'
    ## For image visualization
    image.save(saving_path)
    encoding = processor(images=image, text=question, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    print("Actual Answer: ", answer)

def label_count_list(labels):
    res = {}
    keys = set(labels)
    for key in keys:
        res[key] = labels.count(key)
    return res

def evaluate(processor, model, ds, name):
    if name == "vilt":
        total_size = len(ds['image'])
        print("Beginn evaluation, length of dataset:", total_size)
        acc=0
        j = 0
        wups = 0
        score = 0
        ## for dataset after sclicing
        for i in range(total_size):
            # sample = ds[i]
            label = ds['multiple_choice_answer'][i]
            try:
                encoding = processor(images=ds['image'][i], text=ds['question'][i], return_tensors="pt")
            except Exception:
                print("Load error")
                j+=1
                continue
            else:
            
                outputs = model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                pred = model.config.id2label[idx]

                ## For exact match acc
                if label == pred:
                    acc+=1
                
                ## For pred in annotations:
                ## Not consider confidence here
                ## if appear >= 3 times: score=1
                ## if not appear, score = 0 
                answers = ds["answers"][i]
                ans_list = []
                for answer in answers:
                    ans_list.append(answer["answer"])
                if pred in ans_list:
                    ans_count = label_count_list(ans_list)
                    score += min(1.0, ans_count[pred]/3)
                ## For WUPS, using wordnet; here only consider the first meaning
                ## and some labels are not in wordnet
                syn_label = wordnet.synsets(label)
                syn_pred = wordnet.synsets(pred)
                ## load first synset, first check if it's empty
                if bool(syn_label) and bool(syn_pred):
                    syn_label = syn_label[0]
                    syn_pred = syn_pred[0]
                    wups += syn_label.wup_similarity(syn_pred)
                elif ~bool(syn_label) and ~bool(syn_pred):
                    ## Or also discard them similar to the load error
                    if label == pred:
                        wups += 1

                
        acc /= (total_size-j)
        wups /= (total_size-j)
        score /=(total_size-j)
        print("Accuracy:", acc)
        print("WUPS:", wups)
        print("Score:", score)
        print("error time:", j)
        print("End evaluation")
        return
    if name == "GIT":
        total_size = len(ds['image'])
        print("Beginn evaluation, length of dataset:", total_size)
        acc=0
        j = 0
        wups = 0
        ## for dataset after sclicing
        for i in range(total_size):
            # sample = ds[i]
            label = ds['multiple_choice_answer'][i]
            try:
                pixel_values = processor(images=ds['image'][i], return_tensors="pt").pixel_values
                input_ids = processor(text=ds['question'][i], add_special_tokens=False).input_ids
                input_ids = [processor.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0)
            except Exception:
                print("Load error")
                j+=1
                continue
            else:
            
                generate_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
                ## Output together with question
                output = processor.batch_decode(generate_ids, skip_special_tokens=True)

                output = output[0]
                pred = output.split('?')[-1]
                pred = pred.strip()

                if label == pred:
                    acc+=1
                syn_label = wordnet.synsets(label)
                syn_pred = wordnet.synsets(pred)
                ## load first synset, first check if it's empty
                if bool(syn_label) and bool(syn_pred):
                    syn_label = syn_label[0]
                    syn_pred = syn_pred[0]
                    wups += syn_label.wup_similarity(syn_pred)
                elif ~bool(syn_label) and ~bool(syn_pred):
                    ## Or also discard them similar to the load error
                    if label == pred:
                        wups += 1

                
        acc /= (total_size-j)
        wups /= (total_size-j)
        print("Accuracy:", acc)
        print("WUPS:", wups)
        print("error time:", j)
        print("End evaluation")
        return

evaluate(processor, model, valid_dataset[:5000], "vilt")