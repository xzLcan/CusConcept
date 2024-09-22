# CusDiff
![License](https://img.shields.io/github/license/haoosz/ConceptExpress?color=lightgray)

This is the official PyTorch codes for the paper:  

[**CusDiff: Customized Visual Concept Decomposition with Diffusion Models**]

## Set-up
Create a conda environment `CUS` using
```
conda env create -f environment.yml
conda activate CUS
```

## Extracting CLIP text embeddings

To avoid computing the CLIP text embeddings for the entire vocabulary each time we optimize a decomposition, this code is to extract the CLIP embedding for the entire vocabulary once, and save the embeddings such that you can load them for each concept.

```
python save_dictionary_embeddings.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --path_to_encoder_embeddings="./clip_text_encoding.pt"
```

## Training
Create a new folder that contains an image. For example, download [our dataset](https://drive.google.com/drive/folders/1XvPE-UOwkYM7gVVTj9PlcoTQPKKaRbLn?usp=drive_link) and put it under the root path. You can specify any attribute axis and query the LLM to obtain the corresponding attribute vocabulary, then store it. You can change `--train_data_dir` to the image path and change `vocabulary_path` to the vocabulary path in bash file `scripts/run.sh`. You can specify `--output_dir` to save the checkpoints and generated images. 

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
