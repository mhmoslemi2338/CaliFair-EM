import pandas as pd
import numpy as np
import getopt
import torch
import deepmatcher as dm
import deepmatcher.optim as optim
from HierMatcher import *
import os
import dill

gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


path_base = 'DATA'
for task in [
    'Walmart-Amazon',
    'Beer',
    'Amazon-Google',
    'Fodors-Zagats',
    'iTunes-Amazon',
    'DBLP-GoogleScholar',
    'DBLP-ACM']:
        
        

    dataset_dir = os.path.join(path_base, task) + '/'


    embedding_dir = "embedding"
    


    if not os.path.exists(os.path.join(embedding_dir,"wiki.en.bin")):
        print('please downlaod the embedding file from https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip and rename it to wiki.en.bin and put it in the embedding folder')
        exit()




    train_file = "train.csv"
    valid_file = "valid.csv"
    test_file = "test.csv"
    datasets = dm.data.process(path=dataset_dir,
                                train=train_file,
                                validation=valid_file,
                                test=test_file,
                                embeddings_cache_path=embedding_dir)

    train, validation, test = datasets[0], datasets[1], datasets[2] if len(datasets)>=3 else None


    model = HierMatcher(hidden_size=150,
                        embedding_length=300,
                        manualSeed=2)

    N = 30
    if task =='Beer':
        N = 150
    if task == 'iTunes-Amazon':
        N = 200
    
    model.run_train(train,
                    validation,
                    epochs=N,
                    batch_size=64,
                    label_smoothing=0.05,
                    pos_weight=1.5,
                    best_save_path='best_model_'+dataset_dir.strip('/').split('/')[-1]+'.pth' + gpu_no + '.pth')




    model = HierMatcher(hidden_size=150,
                        embedding_length=300,
                        manualSeed=2)



    model.initialize(train)
    state = torch.load('best_model_'+dataset_dir.strip('/').split('/')[-1]+'.pth1.pth', pickle_module=dill,map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])

    y_score = model.run_prediction(test)
    y_score.to_csv('SCORES/score_HierMatcher_'+task+'.csv', index=False)  




