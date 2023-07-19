import sys
sys.path.append(".")

import streamlit as st
import pandas as pd

from vqa_demo.model_loader import *


# load dataset
ds = load_dataset("test")

# define selector
model_name = st.sidebar.selectbox(
    "Select a model: ",
    ('vilt', 'git', 'blip', 'vbert')
)

image_selector_unspecific = st.number_input(
    "Select an image id: ",
    0, len(ds)
)

# select and display
sample = ds[image_selector_unspecific]
image = sample['image']
image

# inference
question = st.text_input(f"Ask the model a question related to the image: \n"
                               f"(e.g. \"{sample['question']}\")")
args = load_model(model_name) # TODO: cache
answer = get_answer(args, image, question, model_name)
st.write("answer")