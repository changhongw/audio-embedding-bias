# Bias Correction with Pre-trained Audio Embeddings

Implementation of the bias correction methods for pretrained audio embeddings proposed in the following paper:

Changhong Wang, Brian McFee, and Gaël Richard. "Transfer Learning and Bias Correction with Pre-trained Audio Embeddings". _Proceedings of the [International Society for Music Information Retrieval (ISMIR) Conference](https://ismir2023.ismir.net/)_, 2023. 

## Content

- [Installation](#installation)
- [Datasets](#datasets)
- [Pre-trained embeddings](#pre-trained-embeddings)
- [Bias correction](#bias-correction)
- [Contact](#contact)
- [Cite](#cite)

## Installation
We recommend using Conda environment:

```sh
git clone https://github.com/changhongw/deem.git
conda env create -f environment.yml
conda activate deem
```

## Datasets

Download [IRMAS](https://www.upf.edu/web/mtg/irmas) and [OpenMIC](https://zenodo.org/record/1432913) datasets and save as directories `./data/irmas` and `./data/openmic-2018`, respectively.

## Pre-trained embeddings

Extracted [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), [OpenL3](https://github.com/marl/openl3), and [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) embeddings for both datasets. Or use our [extracted pre-trained embeddings]() directly.

## Bias correction

Run the note books in `notebooks`:
- `0_data_distribution.ipynb`: investigate the distribution of each dataset in terms of genre distribution and number of samples per class
- `1_debias_linear.ipynb`: linear bias correction (original, LDA, mLDA)
- `2_debias_nonlinear.ipynb`: linear bias correction (K, KLDA, mKLDA)
- `3_cosine_similarity.ipynb`: calculate cosine similarity between dataset separation and instrument classification
- `4_performance_summary.ipynb`: summarize results from all bias correction methods

## Contact

For any questions, support, or inquiries, please feel free to contact [changhong.wang@telecom-paris.fr](mailto:changhong.wang@telecom-paris.fr).

## Cite

Please cite the following paper if you use the code provided in this repository.

 > Changhong Wang, Brian McFee, and Gaël Richard. "Transfer Learning and Bias Correction with Pre-trained Audio Embeddings". _Proceedings of the International Society for Music Information Retrieval (ISMIR) Conference_, 2023. 

```bibtex
@inproceedings{wang2023bias,
    author = {Changhong Wang and Brian McFee and Gaël Richard},
    title = {Transfer Learning and Bias Correction with Pre-trained Audio Embeddings},
    booktitle = {Proceedings of the International Society for Music Information Retrieval (ISMIR) Conference},
    year = 2023,
}
```
