# Transfer learning and bias correction with pre-trained audio embeddings

## How to run
- clone the repository
- add the pre-trained embeddings `embeddings.h5` and openmic-2018 dataset `openmic-2018` to the directly
- run `1_dataset_separate_extraction.ipynb` to extract the dataset separation direction vector
- run `2_bias_correction_all_experiments.ipynb` for within- and cross-domain classification using embeddings before and after debiasing
- run `3_plot_results.ipynb` to plot all the result, including cosine similarity of dataset separation and instrument separation before and after kernelization, genre distribution, and different visalization of the debiasing results