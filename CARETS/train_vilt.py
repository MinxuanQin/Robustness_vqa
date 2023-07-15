## Author: Minxuan Qin
## Changes in this script: rearange dataset and dataloader, consider VQA as a classification task for fine-tuning
## Here it is an example of fine-tuning on negation dataset spilt and I use 40% data for fine-tuning.
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image

from carets import CaretsDataset
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
from transformers import AutoProcessor, AutoModelForCausalLM

import os
import wandb
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
orig_dataset = CaretsDataset('configs/default.yml')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", num_labels=len(config.id2label), id2label=config.id2label, label2id=config.label2id)

# processor = AutoProcessor.from_pretrained("microsoft/git-base-vqav2", padding_side='left')
# model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")
model.to(device)

def load_and_save_carets_path(test_name, split):
    input_data = []
    for question in split:
        ques = dict()
        question_id = question['question_id']
        img_path = question['image_path']
        question_sent = question['sent']
        ## add label
        label = question['label']
        if list(label.keys())[0] not in config.label2id.keys():
            continue

        ques['image'] = img_path
        ques['question'] = question_sent
        ques['question_id'] = question_id
        ques['score'] = label
        input_data.append(ques)
    np.save(f'data/vilt_{test_name}.npy', input_data)

class Caret_Torch_Dataset(Dataset):
    def __init__(self, test_name):
        self.data = np.load(f'data/vilt_{test_name}.npy', allow_pickle=True)
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
        output_item['ans_id'] = config.label2id[list(single_item['score'].keys())[0]]
        output_item['ans_score'] = list(single_item['score'].values())[0]
        try:
            encoding = processor(images=img, text=output_item['ques'], padding="max_length", truncation=True, return_tensors="pt")

            for k,v in encoding.items():
                encoding[k] = v.squeeze()

            targets = torch.zeros(len(config.id2label))
            targets[output_item['ans_id']] = output_item['ans_score']
            encoding["labels"] = targets
            encoding["question_id"] = output_item['id']

        except Exception:
            # print('load error')
            output_item['test_type'] = self.test_name
            output_item['img_path'] = img_path

            return None
        
        return encoding
    
    def __len__(self):
        return len(self.data)

def my_collate_fn(batch):
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0: return dict()
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    ids = [item['question_id'] for item in batch]
    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)
    batch['q_ids'] = ids
    
    
    return batch

img_path = []
for test_name, split in orig_dataset.splits:
    ## wandb
    
    wandb.init(project=f"carets-{test_name}",
    config={
        "learing_rate": 5e-5, 
        "dataset": f"{test_name}",
        "epochs": 10,
            })

    load_and_save_carets_path(test_name, split)
    if test_name == 'negation_consistency':
        epochs = 10
        dataset = Caret_Torch_Dataset(test_name)
        train_size = int(len(dataset)*0.4)
        valid_size = len(dataset) - train_size
        train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
        train_dataloader = DataLoader(dataset=train_set,
        batch_size=16, shuffle=True, collate_fn=my_collate_fn, num_workers=1)
        valid_dataloader = DataLoader(dataset=dataset,
        batch_size=16, shuffle=False, collate_fn=my_collate_fn, num_workers=1)

        special_loader = DataLoader(dataset=valid_set,
        batch_size=16, shuffle=False, collate_fn=my_collate_fn, num_workers=1)

        print('Begin of the test: ', test_name)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            t_begin = time.perf_counter()
            model.train()
            total_len = 0
            for batch in train_dataloader:

                q_ids = batch.pop('q_ids')

                batch = {k:v.to(device) for k,v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                wandb.log({"loss": loss})
                loss.backward()
                optimizer.step()
            
            model.eval()
            predictions = dict()
            for eval_batch in valid_dataloader:
                q_ids = eval_batch.pop('q_ids')
                eval_batch = {k:v.to(device) for k,v in eval_batch.items()}
                output = model(**eval_batch)
                idx = output.logits.argmax(dim=1).cpu()
                ## write predictions
                # total_len += len(q_ids)
                for i, id in enumerate(q_ids):
                    predictions[id] = model.config.id2label[int(idx[i])]

            print("Complete predictions: ", len(predictions))
            t_end = time.perf_counter()

            predictions1 = dict()
            for eval_batch in special_loader:
                q_ids = eval_batch.pop('q_ids')
                eval_batch = {k:v.to(device) for k,v in eval_batch.items()}
                output = model(**eval_batch)
                idx = output.logits.argmax(dim=1).cpu()
                ## write predictions
                # total_len += len(q_ids)
                for i, id in enumerate(q_ids):
                    predictions1[id] = model.config.id2label[int(idx[i])]
            
            print("Running Time: %s min"%((t_end-t_begin)/60))
            np.save(f"vilt_res1/{test_name}_{epoch}.npy", predictions)   
            np.save(f"vilt_res1/{test_name}_{epoch}_1.npy", predictions1)         
            model.save_pretrained('/cluster/scratch/minqin/hf_vqa/CARETS/vilt_neg_model')  
    

    


