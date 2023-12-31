# Instructions on dataset construction
You can find the original repository [here](https://github.com/princeton-nlp/CARETS).
## Download image files
All the images used in CARETS come from the [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) *validation* set. If you've already downloaded the GQA dataset, you may set the ```images_root``` element in the dataset config to be the GQA images directory. Otherwise, you have two options: 1) download all the images for the GQA dataset [from here](https://cs.stanford.edu/people/dorarad/gqa/download.html) (20GB) or 2) download just the subset of images that we use with the script below (1.3GB).
```bash
cd CARETS

export DATADIR=data  # where to store images directory
export TARNAME=images.tar.gz

wget --save-cookies pbbxvrf.txt 'https://drive.google.com/uc?id=1Yi_Zgbn0rraekBV96Vwmg9kOuv72b1Lt&export=download' -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > pbasvez.txt && \
wget --load-cookies pbbxvrf.txt -O $TARNAME \
     'https://drive.google.com/uc?id=1Yi_Zgbn0rraekBV96Vwmg9kOuv72b1Lt&export=download&confirm='$(<pbasvez.txt) && \
tar -xzf $TARNAME -C $DATADIR

rm -f pbbxvrf.txt pbasvez.txt $TARNAME
```

## Dataset configuration
After image downloading, you can find the configuration file under `configs/default.yml`, it contains five tests from the CARETS paper.

## Changes in CaretsDataset
We fix minor bugs of original and perturbed accuracy computation in the original code and add additional metrics in terms of WUPS. For changes see `carets/dataset.py`.

## Predictions generation
Before sending data to the CaretsDataset, we need predictions of all questions within the dataset from the pre-trained model. Minxuan wrote all scripts for the evaluation and fine-tuning process. Our main contribution is:

1. Define a custom Dataset and Dataloader to read and filter data from CaretsDataset, which also support batch processing running on a single GPU. There exist some invalid data (less than 0.05%) because the image contains only one channel.

2. Finish the loading script for [ViLT](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/vilt#transformers.ViltForQuestionAnswering), [GIT](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/git#transformers.GitForCausalLM), [VisualBERT](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering), and [BLIP](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/blip#transformers.BlipForQuestionAnswering) from the Hugging Face platform; each of them need specific adaption for prediction generation. (see `inf_\*.py`)

## Performance evaluation on CARETS dataset
See `carets_eval.py`.

## Fine-tuning pre-trained models
For efficient training Minxuan rearange the custom Dataset and Dataloader definition and train the ViLT model on part of the CaretsDataset by condifering VQA as a classification task. For the training script see `train_eval.py` and for the evaluation of fune-tuned model see `eval_vilt.py`.

## Dataset extraction for analysis
Minxuan writes some help functions for data extraction, which is needed for the dataset analysis. (see `data_analysis.py`.)

## Citation
Kudos to the authors for their amazing results:
```bibtex
@inproceedings{jimenez2022carets,
   title={CARETS: A Consistency And Robustness Evaluative Test Suite for VQA},
   author={Carlos E. Jimenez and Olga Russakovsky and Karthik Narasimhan},
   booktitle={60th Annual Meeting of the Association for Computational Linguistics (ACL)},
   year={2022}
}
```
