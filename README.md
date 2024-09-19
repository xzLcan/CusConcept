# DOCS
![License](https://img.shields.io/github/license/haoosz/ConceptExpress?color=lightgray)
DOCS: Decomposing Generative Visual Concepts under Human Specification

## Set-up
Create a conda environment `DOCS` using
```
conda env create -f environment.yml
conda activate DOCS
```

## Extracting CLIP text embeddings

To avoid computing the CLIP text embeddings for the entire vocabulary each time we optimize a decomposition, this code to extract the CLIP embedding for the entire vocabulary once, and save the embeddings such that you can load them for each concept.

```
python -m save_dictionary_embeddings.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --path_to_encoder_embeddings="./clip_text_encoding.pt"
```

## Training
Create a new folder that contains an image. For example, download [our dataset](https://drive.google.com/drive/folders/1XvPE-UOwkYM7gVVTj9PlcoTQPKKaRbLn?usp=drive_link) and put it under the root path. You can change `--train_data_dir` in bash file `scripts/run.sh` to the image path. You can specify `--output_dir` to save the checkpoints. 

When the above is ready, run the following to start training:
```
bash scripts/run.sh
```

## Citation
If you use this code in your research, please consider citing our paper:
```bibtex
```

## Acknowledgements
This code repository is based on the great work of [Conceptor](https://github.com/hila-chefer/Conceptor). Thanks!