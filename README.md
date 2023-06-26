# Eembedding Bias Correction

## Preparation

### Intall requirements
git clone https://github.com/changhongw/examod.git
conda env create -f environment.yml
conda activate embedding-bis-correction

### Download dataset
[https://www.upf.edu/web/mtg/irmas]IRMAS dataset and [https://zenodo.org/record/1432913]OpenMIC dataset

### Pre-trained embedding extraction
[https://github.com/tensorflow/models/tree/master/research/audioset/vggish]VGGish, [https://github.com/marl/openl3]OpenL3, and [https://github.com/tensorflow/models/tree/master/research/audioset/yamnet]YAMNet

## Bias correction
Run the note books in `scripts`:
- `0_data_distribute.ipynb` :
- `1_debias_linear.ipynb`
- `2_debias_nonlinear.ipynb`
- `3_cosine_similarity.ipynb`
- `4_performance_change_linear.ipynb`
- `4_performance_change_nonlinear.ipynb`

## Cite
Changhong Wang, Brian McFee, and Gaël Richard. "Transfer Learning and Bias Correction with Pre-trained Audio Embeddings". International Society for Music Information Retrieval (ISMIR) conference, 2023.
