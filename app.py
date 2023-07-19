import numpy as np
from PIL import Image
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
import cv2
import streamlit as st

st.title("Live demo of multimodal vqa")

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("Minqin/carets_vqa_finetuned")

orig_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

uploaded_file = st.file_uploader("Please upload one image", type=["jpg", "png", "bmp", "jpeg"])

question = st.text_input("Type here one question on the image")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    image_cv2 = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    st.image(image_cv2, channels="RGB")

    img = Image.fromarray(image_cv2)

    encoding = processor(images=img, text=question, return_tensors="pt")

    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    pred = model.config.id2label[idx]

    orig_outputs = orig_model(**encoding)
    orig_logits = orig_outputs.logits
    idx = orig_logits.argmax(-1).item()
    orig_pred = orig_model.config.id2label[idx]
    st.text(f"Answer of ViLT: {orig_pred}")
    st.text(f"Answer after fine-tuning: {pred}")
