# Exploring robustness and consistency of multimodal VQA models
course project of CSNLP at ETH, FS 2023, hugging face demo see [link](https://huggingface.co/spaces/Minqin/carets_finetune_vqa).

## Set up
For running on ETH Euler cluster, please run first `env_setup.sh` to install required packages. If needed, `hf_env_script.sh` contains definitions of environment variables for Hugging Face cache management.


For setting up CARETS, please read first `README.md` under `CARETS` directory. The directory also contains an example script `run_eval.sh` for running on ETH Euler cluster.

## Baseline evaluation
See python scripts under `baseline` directory.

## Dataset exploration and visualization
The `json` file under `stats` directory contains all questions from the CARETS dataset. We also provide the code for the visualization plot of the performance on CARETS under `visualization`.

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
