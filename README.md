# Food Recipe Recommender using Neural Matrix Factorization
Food lovers and home cooks are constantly seeking inspiration and guidance to elevate their culinary journeys. A recipe recommender system can offer tailored suggestions based on individual preferences, dietary restrictions, and available ingredients. By catering to the unique needs of users, the application can provide immense value by enhancing their cooking experiences and expanding their culinary horizons. In this project, we aim to build this recommender in neural network based matrix factorization.

## Dataset
The dataset used is “Food.com Recipes and Interactions” consists of 180K+ recipes and 700K+ recipe reviews covering 18 years of user interactions and uploads on Food.com, obtained from Kaggle: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions. RAW_recipies.csv and RAW_interactions.csv files were from the dataset. 

## Architecture
The model architecture is inspired from Neural Collaborative Filtering paper https://arxiv.org/abs/1708.05031


## Required Packages
- tqdm: for progress bar
- matplotlib: for plotting
- torch (PyTorch): for deep learning
- Pillow: for image loading
- pandas: for csv data handling
- scikit-learn: to provide various metric functions


### Install
#### Using Nvidia GPU (Cuda 11.8)
```
pip3 install tqdm matplotlib Pillow scikit-learn pandas torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Using CPU Only: 
### Although, you can use CPU Only, it would take longer time to train
```
pip3 install tqdm matplotlib Pillow scikit-learn pandas torch 
```

## Directory Structure
Place all the files in same directory as the following:
```
├─── data/          contains csv data files
├─── datasets/      contains custom dataset classes
├─── models/        contains recommender models
├─── utils/         contains helper functions
├─── results/       contains metrics results
├─── NeuralMF.pt    trained model weights for NeuralMatrixFactorization
├─── demo.ipynb     jupyter notebook run the demo 
├─── train.py       for train the recommender model
└─── recommender.py for recommendation functions
```

## Running Demo
To run the demo, please run the following Jupyter Notebook: demo.ipynb

** Recommend using VSCode https://code.visualstudio.com for running the demo notebook