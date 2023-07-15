## Author: Minxuan Qin
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image

from carets import CaretsDataset
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForQuestionAnswering

import time

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
orig_dataset = CaretsDataset('configs/default.yml')
predictions = dict()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

error_data = []

#processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2", padding_side='left')
# model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")

processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')
model.to(device)

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

class Caret_Torch_Dataset(Dataset):
    def __init__(self, test_name):
        self.data = np.load(f'data/{test_name}.npy', allow_pickle=True)
        self.length = len(self.data)
        self.test_name = test_name

    def __getitem__(self, index):
        single_item = self.data[index]
        output_item = dict()

        img_path = single_item['image']
        img = Image.open(img_path)
        # img = transform1(img)

        output_item['img'] = img
        # output_item['ques'] = single_item['question']
        ## GIT, add cls token
        output_item['ques'] = processor.tokenizer.cls_token + single_item['question']
        output_item['id'] = single_item['question_id']
        try:
            ## Vilt
            # encoding = processor(images=img, text=output_item['ques'], return_tensors="pt")
            
            ## GIT
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            input_ids = processor(text=output_item['ques'], add_special_tokens=False).input_ids
            input_ids = [processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            
        except Exception:
            # print('load error')
            output_item['test_type'] = self.test_name
            output_item['img_path'] = img_path
            error_data.append(output_item)
            return None

        return output_item
    
    def __len__(self):
        return len(self.data)

def my_collate_fn(batch):
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0: return dict()
    return {
        'img': [x['img'] for x in batch],
        'ques': [x['ques'] for x in batch],
        'id': [x['id'] for x in batch]
    }

img_path = []
test_i = 0
for test_name, split in orig_dataset.splits:
    # load_and_save_carets_path(test_name, split)
    dataset = Caret_Torch_Dataset(test_name)
    dataloader = DataLoader(dataset=dataset,
    batch_size=8, shuffle=True, collate_fn=my_collate_fn, num_workers=1)
    
    print('Begin of the test: ', test_name)
    t_begin = time.perf_counter()
    total_len = 0
    for batch in dataloader:
        question_ids = batch['id']
        
        ## BLIP
        pixel_values = processor(images=batch['img'], return_tensors="pt").pixel_values
        batch_input_ids = processor(text=batch['ques'], add_special_tokens=False, padding=True).input_ids
        ## already added cls_token for each entry in dataset
        batch_input_ids = torch.tensor(batch_input_ids)
        batch_input_ids = batch_input_ids.to(device)
        pixel_values = pixel_values.to(device)
        ## generate output
        generate_ids = model.generate(pixel_values=pixel_values, input_ids=batch_input_ids, max_length=50)
        output = processor.batch_decode(generate_ids.cpu(), skip_special_tokens=True)

        total_len += len(question_ids)
        for i, id in enumerate(question_ids):
            # sent = output[i]
            pred = output[i]
            ## For Blip: output does not include question
            #pred = sent.split('?')[-1]
            #pred = pred.strip()
            predictions[id] = pred
        '''

        encoding = processor(images=batch['img'], text=batch['ques'], return_tensors='pt', padding=True)
        encoding.to(device)
        output = model(**encoding)
        idx = output.logits.argmax(dim=1).cpu()
        ## write predictions
        total_len += len(question_ids)
        for i, id in enumerate(question_ids):
            predictions[id] = model.config.id2label[int(idx[i])]
        '''
    t_end = time.perf_counter()
    print("Running Time: %s min"%((t_end-t_begin)/60))
    print("Inference time for each sample: %s s"%((t_end-t_begin) / total_len))
    print('Error size: ', len(dataset) - total_len)
    test_i += 1

np.save('preds_blip.npy', predictions)
np.save('error_data_blip.npy', error_data)

    


