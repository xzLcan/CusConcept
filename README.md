# CusConcept
![License](https://img.shields.io/badge/license-MIT-lightgray)
[![arXiv](https://img.shields.io/badge/arXiv-2410.00398%20-b31b1b)](https://arxiv.org/abs/2410.00398)


This is the official PyTorch codes for the paper:  

[**CusConcept: Customized Visual Concept Decomposition with Diffusion Models**](https://arxiv.org/abs/2410.00398)  
<a href="https://www.linkedin.com/in/zhi-xu-03764a28b/">Zhi Xu</a><sup>\*</sup>,
<a href="https://haoosz.github.io/">Shaozhe Hao</a><sup>\*</sup>,
<a href="https://www.kaihan.org/">Kai Han</a><sup>†</sup>  
*WACV 2025*
## Set-up
Create a conda environment `CUS` using:
```
conda env create -f environment.yml
conda activate CUS
```

## Extracting CLIP text embeddings

To avoid computing the CLIP text embeddings for the entire vocabulary each time, this code is to extract the CLIP embedding for the entire vocabulary once, and save the embeddings such that you can load them for each concept.

```
python save_dictionary_embeddings.py
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
@misc{xu2024cusconceptcustomizedvisualconcept,
      title={CusConcept: Customized Visual Concept Decomposition with Diffusion Models}, 
      author={Zhi Xu and Shaozhe Hao and Kai Han},
      journal={arXiv preprint arXiv:2410.00398},
      year={2024}
}
```

## Acknowledgements
This code repository is based on the great work of [Conceptor](https://github.com/hila-chefer/Conceptor). Thanks!
