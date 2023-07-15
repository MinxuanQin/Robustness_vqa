## Author: Minxuan Qin
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image

from carets import CaretsDataset
from torchvision.models import resnet50
import torchvision.transforms as transforms
from transformers import VisualBertForMultipleChoice, VisualBertForQuestionAnswering, BertTokenizerFast, AutoTokenizer, ViltForQuestionAnswering
import time
import os
import requests
import timm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
orig_dataset = CaretsDataset('configs/default.yml')
predictions = dict()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
model.eval()
img_model_name = "resnet50"

def load_img_model(name):
    """
    loads image models for feature extraction
    returns model name and the loaded model
    """
    if name == "resnet50":
        model = resnet50(weights='DEFAULT')
    elif name == "vitb16":
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    else:
        raise ValueError("undefined model name: ", name)

    return model

# helper functions for single item
def get_single_item(ques, img, tokenizer, model_name, image_model):
    inputs = tokenizer(
        ques,
        # padding='max_length',
        # truncation=True,
        # max_length=128,
        return_tensors='pt',
    )
    visual_embeds = get_img_feats(img, image_model=image_model, name=model_name)
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

    return inputs

# helper functions
def get_item(ques, img, tokenizer, model_name, image_model):
    ## input img: 3d/4d tensor, ques: list of str
    if img.dim() == 3:
        return get_single_item(ques, img, tokenizer, model_name, image_model)
    inputs = tokenizer(
        ques,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    visual_embeds = get_img_feats(image=img, image_model=image_model, name=model_name)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    ## shape of visual_embeds: (B, visual_seq_len, feature_dim)
    upd_dict = {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }

    inputs.update(upd_dict)

    return inputs


def get_img_feats(image, image_model, name='resnet50'):
    ## image: tensors
    if image.dim() < 4:
        image = image.unsqueeze(0)
    if name == "resnet50":
        image_model = torch.nn.Sequential(*list(image_model.children())[:-1])

    if name == "resnet50":
        image_features = image_model(image)
        image_features = image_features.squeeze(2,3)
    elif name == "vitb16":
        ## MOD Minxuan: (B, 768), but does not work, becuase the model need a 2048 feature vector
        # image_features = image_model.forward_features(image)
        image_features = image_model(image)

    ## shape (B, 1, feature_dim); 1 = visual_seq_len
    image_features = image_features.unsqueeze(1)
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


model.to(device)
## For vilt, reshape to 224, 224; for resnet50 also works
transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),  # Convert PIL Image back to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_and_save_carets_path(test_name, split):
    input_data = []
    for question in split:
        ques = dict()
        question_id = question['question_id']
        img_path = question['image_path']
        question_sent = question['sent']

        ques['image'] = img_path
        ques['question'] = question_sent
        ques['question_id'] = question_id
        input_data.append(ques)
    np.save(f'data/{test_name}.npy', input_data)

error_data = []
image_model = load_img_model(img_model_name)
image_model.eval()

## Get vqa id->answer for VisualBert
vqa_answers = get_data(VQA_URL)

class Caret_Torch_Dataset(Dataset):
    def __init__(self, test_name):
        self.data = np.load(f'data/{test_name}.npy', allow_pickle=True)
        self.length = len(self.data)
        self.test_name = test_name

    def __getitem__(self, index):
        single_item = self.data[index]
        output_item = dict()

        img_path = single_item['image']
        img_pil = Image.open(img_path)
        
        output_item['ques'] = single_item['question']
        output_item['id'] = single_item['question_id']
        try:
            img = transform(img_pil)

            output_item['img'] = img
            inputs = get_item(output_item['ques'], img, tokenizer, img_model_name, image_model)
        except Exception:
            output_item['test_type'] = self.test_name
            output_item['img'] = img_pil
            error_data.append(output_item)
            return None

        return output_item
    
    def __len__(self):
        return len(self.data)

def my_collate_fn(batch):
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0: return dict()
    img_tensors = [x['img'] for x in batch]
    return {
        'img': torch.stack(img_tensors),
        'ques': [x['ques'] for x in batch],
        'id': [x['id'] for x in batch]
    }

for test_name, split in orig_dataset.splits:
    # load_and_save_carets_path(test_name, split)
    dataset = Caret_Torch_Dataset(test_name)
    dataloader = DataLoader(dataset=dataset,
    batch_size=8, shuffle=True, collate_fn=my_collate_fn, num_workers=1)

    print('Begin of the test: ', test_name)
    t_begin = time.perf_counter()
    total_len = 0
    for batch in dataloader:
        if len(batch) == 0: continue
        question_ids = batch['id']
        ques = batch['ques']
        imgs = batch['img']
        input_encoding = get_item(ques, imgs, tokenizer, img_model_name, image_model)
        input_encoding = input_encoding.to(device)
        outputs = model(**input_encoding)

        pred_idx = outputs.logits.argmax(dim=1).cpu()
        ## write predictions
        total_len += len(question_ids)
        for i, id in enumerate(question_ids):
            predictions[id] = vqa_answers[int(pred_idx[i])]
    t_end = time.perf_counter()
    print("Running Time: %s min"%((t_end-t_begin)/60))
    print("Inference time for each sample: %s s"%((t_end-t_begin) / total_len))
    print('Error size: ', len(dataset) - total_len)

np.save('preds_visual_resnet.npy', predictions)
np.save('error_data_visual_resnet.npy', error_data)
    
    

