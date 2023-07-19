from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from datasets import load_dataset, get_dataset_split_names
import numpy as np

import requests
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import AutoProcessor, AutoModelForCausalLM
from nltk.corpus import wordnet

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

# load processor and model
def load_model(name):
    if name == "vilt":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained("CARETS/vilt_neg_model")
    elif name == "git":
        processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")
    elif name == "blip":
        pass
    elif name == "vbert":
        processor = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
    else:
        raise ValueError("invalid model name: ", name)

    return (processor, model)


def load_dataset(type):
    if type == "train":
        return load_dataset("HuggingFaceM4/VQAv2", split="train", cache_dir="cache", streaming=False)
    elif type == "test":
        return load_dataset("HuggingFaceM4/VQAv2", split="validation", cache_dir="cache", streaming=False)
    else:
        raise ValueError("invalid dataset: ", type)


def tokenize_function(examples, processor):
    sample = {}
    sample['inputs'] = processor(images=examples['image'], text=examples['question'], return_tensors="pt")
    sample['outputs'] = examples['multiple_choice_answer']
    return sample


def label_count_list(labels):
    res = {}
    keys = set(labels)
    for key in keys:
        res[key] = labels.count(key)
    return res


def get_item(image, question, tokenizer, image_model, model_name):
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

def err_msg():
    print("Load error, try again")
    return "[ERROR]"


def get_answer(model_loader_args, img, question, model_name):
    processor, model = model_loader_args[0], model_loader_args[1]
    if model_name == "vilt":
        try:
            encoding = processor(images=img, text=question, return_tensors="pt")
        except Exception:
            return err_msg()
        else:
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            pred = model.config.id2label[idx]

    elif model_name == "git":
        try:
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            input_ids = processor(text=question, add_special_tokens=False).input_ids
            input_ids = [processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        except Exception:
            return err_msg()
        else:
            generate_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
            output = processor.batch_decode(generate_ids, skip_special_tokens=True)
            output = output[0]
            pred = output.split('?')[-1]
            pred = pred.strip()

    elif model_name == "vbert":
        vqa_answers = get_data(VQA_URL)
        try:
            # load question and image (processor = tokenizer)
            _, inputs = get_item(img, question, processor, model_name)
            outputs = model(**inputs)
        except Exception:
            return err_msg()
        else:
            answer_idx = torch.argmax(outputs.logits, dim=1).item()  # from 3129
            pred = vqa_answers[answer_idx]

    elif model_name == "blip":
        pass

    else:
        return "Invalid model name"

    return pred
