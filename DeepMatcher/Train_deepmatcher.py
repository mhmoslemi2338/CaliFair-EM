import deepmatcher as dm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os


gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no




N_epochs = 30
path_base = 'DATA'


for task in [
    'DBLP-GoogleScholar',
    'iTunes-Amazon',
    'Beer',
    'Walmart-Amazon',
    'Amazon-Google',
    'Fodors-Zagat',
    'DBLP-ACM']:


    dataset_dir = os.path.join(path_base, task) +"/"




    embedding_dir = '../Hiermathcer/embedding'

    if not os.path.exists(os.path.join(embedding_dir,"wiki.en.bin")):
        print('please downlaod the embedding file from https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip and rename it to wiki.en.bin and put it in the embedding folder')
        exit()



    model_path = 'best_model_'+task+'.pth'

    train_file = "train.csv"
    valid_file = "valid.csv"
    test_file = "test.csv"

    train, validation, test = dm.data.process(path=dataset_dir,
        train=train_file, validation=valid_file, test=test_file, embeddings_cache_path=embedding_dir)




    model = dm.MatchingModel()
    model.run_train(train, validation, best_save_path=model_path, epochs = N_epochs)



    model = dm.MatchingModel()
    model.load_state(model_path)


    pred_y = model.run_prediction(test)
    pred_y.to_csv('../SCORES/score_DeepMatcher_'+task+ '.csv', index=False) 


