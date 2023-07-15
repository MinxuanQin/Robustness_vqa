"""
Partially based on
https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing
"""
## Authors: Han Xi, Minxuan
## only works for resnet50 backbone bacause vitb16 backbone has incompatible feature dimension
from builtins import breakpoint
import os
import requests
from tqdm import tqdm
import timm

# VLMO: modify in vlmo/config.py: set test_only -> True
from datasets import load_dataset, get_dataset_split_names

import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
from transformers import VisualBertForMultipleChoice, VisualBertForQuestionAnswering, BertTokenizerFast, AutoTokenizer, ViltForQuestionAnswering

from PIL import Image
from nltk.corpus import wordnet
import time

# global params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"


# helper functions
def get_item(idx, ds, tokenizer, model_name):
    item = ds[idx]
    question = item['question']
    image = item['image']
    # print(question)
    # image.show()

    inputs = tokenizer(
        question,
        # padding='max_length',
        # truncation=True,
        # max_length=128,
        return_tensors='pt'
    )
    visual_embeds = get_img_feats(image, image_model=image_model, name=model_name)\
        .squeeze(2, 3).unsqueeze(0)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    # print("embed shape: ", visual_embeds.shape)
    # print("visual attn shape: ", visual_attention_mask.shape)
    upd_dict = {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
    inputs.update(upd_dict)

    return upd_dict, inputs


def get_img_feats(image, image_model, new_size=None, name='resnet50'):
    if name == "resnet50":
        image_model = torch.nn.Sequential(*list(image_model.children())[:-1])

    # apply transforms when necessary
    if new_size is not None:
        transfrom_f = transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.LANCZOS)
        image = transfrom_f(image)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image back to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get features
    image = transform(image)
    if name == "resnet50":
        image_features = image_model(image.unsqueeze(0))
    elif name == "vitb16":
        image_features = image_model.forward_features(image.unsqueeze(0))
    return image_features


def get_data(query, delim=","):
    assert isinstance(query, str)
    if os.path.isfile(query):
        with open(query) as f:
            data = eval(f.read())
    else:
        req = requests.get(query)
        try:
            data = requests.json()
        except Exception:
            data = req.content.decode()
            assert data is not None, "could not connect"
            try:
                data = eval(data)
            except Exception:
                data = data.split("\n")
        req.close()
    return data

def label_count_list(labels):
    res = {}
    keys = set(labels)
    for key in keys:
        res[key] = labels.count(key)
    return res

def evaluate(tokenizer, model, model_name, ds):
    total_size = len(ds['image'])
    print("Begin evaluation, length of dataset:", total_size)
    acc = 0
    j = 0
    wups = 0
    score = 0
    # for dataset after sclicing
    for i in tqdm(range(total_size)):
        label = ds['multiple_choice_answer'][i]
        try:
            # load question and image
            out_dict, inputs = get_item(i, valid_dataset, tokenizer, model_name)
            outputs = model(**inputs)
        except Exception:
            print("load error")
            j += 1
            continue
        else:
            answer_idx = torch.argmax(outputs.logits, dim=1).item()  # from 3129
            predicted_answer = vqa_answers[answer_idx]
            print("Predicted Answer:", predicted_answer)
            print("Actual answer: ", label)

            if predicted_answer == label:
                acc += 1
            
            answers = ds["answers"][i]
            ans_list = []
            for answer in answers:
                ans_list.append(answer["answer"])
            if predicted_answer in ans_list:
                ans_count = label_count_list(ans_list)
                score += min(1.0, ans_count[predicted_answer]/3)
            ## For WUPS, using wordnet; here only consider the first meaning
            ## and some labels are not in wordnet
            syn_label = wordnet.synsets(label)
            syn_pred = wordnet.synsets(predicted_answer)
            ## load first synset, first check if it's empty
            if bool(syn_label) and bool(syn_pred):
                syn_label = syn_label[0]
                syn_pred = syn_pred[0]
                wups += syn_label.wup_similarity(syn_pred)
            elif ~bool(syn_label) and ~bool(syn_pred):
                ## Or also discard them similar to the load error
                if label == predicted_answer:
                    wups += 1

    acc /= (total_size - j)
    score /= (total_size - j)
    wups /= (total_size - j)
    print("Accuracy:", acc)
    print("Score:", score)
    print("WUPS:", wups)
    print("error time:", j)
    print("End evaluation")


def load_img_model(name):
    """
    loads image models for feature extraction
    returns model name and the loaded model
    """
    if name == "resnet50":
        model = resnet50(weights='DEFAULT')
    elif name == "vitb16":
        ## MOD Minxuan: add param
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    else:
        raise ValueError("undefined model name: ", name)

    return model, name


# evaluation
if __name__ == "__main__":
    # id -> answer
    vqa_answers = get_data(VQA_URL)
    image_models = ["resnet50", "vitb16"]

    # load image and language model
    image_model, model_name = load_img_model("resnet50")
    image_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

    # prepare datasets
    train_dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", cache_dir="cache", streaming=False)
    valid_dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation", cache_dir="cache", streaming=False)

    # eval, note: image features should be of size [BS, 2048]
    # ViT: forward_features: ([B, 197, 768]), FC: ([B, 1000])
    num_samples = 5
    model.to(device)
    model.eval()
    t1 = time.perf_counter()
    evaluate(tokenizer, model, model_name, valid_dataset[:num_samples])#valid_dataset[:num_samples])
    t2 = time.perf_counter()
print("Running time: %s s"%((t2-t1)))
print("Inference time for each sample: %s s" % ((t2-t1)/num_samples))