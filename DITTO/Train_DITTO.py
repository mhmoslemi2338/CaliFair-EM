import os
import json
import sys
import torch
import numpy as np
import random
from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train
from torch.utils import data
from ditto_light.ditto import DittoModel
import pandas as pd




gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no





for task in [
    "Structured/Amazon-Google",
    "Structured/Beer",
    "Structured/DBLP-ACM",
    "Structured/DBLP-GoogleScholar",
    "Structured/Fodors-Zagats",
    "Structured/iTunes-Amazon",
    "Structured/Walmart-Amazon"
]:





    run_id = 0
    batch_size = 64
    max_len = 256
    lr = 1e-5
    n_epochs = 15
    alpha_aug = 0.8
    size = None
    lm = 'distilbert'
    da = None # default=None
    dk = None # default=None




    logdir = "checkpoints/"
    summarize = True
    finetuning = True
    save_model = True
    fp16 = False






    # set seeds
    random.seed(run_id)
    np.random.seed(run_id)
    torch.manual_seed(run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_id)


    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, lm, da,dk, summarize, str(size), run_id)
    run_tag = run_tag.replace('/', '_')


    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]


    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']



    # summarize the sequences up to the max sequence length
    summarizer = Summarizer(config, lm=lm)
    trainset = summarizer.transform_file(trainset, max_len=max_len)
    validset = summarizer.transform_file(validset, max_len=max_len)
    testset = summarizer.transform_file(testset, max_len=max_len)


    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                    lm=lm,
                                    max_len=max_len,
                                    size=size,
                                    da=da)
    valid_dataset = DittoDataset(validset, lm=lm)
    test_dataset = DittoDataset(testset, lm=lm)


    HP = {} 
    HP['n_epochs'] = n_epochs
    HP['batch_size'] = batch_size
    HP['logdir'] = logdir
    HP['n_epochs'] = n_epochs
    HP['save_model'] = save_model
    HP['logdir'] = logdir
    HP['task'] = task
    HP['alpha_aug'] = alpha_aug
    HP['fp16'] = fp16
    HP['lm'] = lm
    HP['lr'] = lr

    train(train_dataset,
            valid_dataset,
            test_dataset,
            run_tag, HP)


    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------



    test_iter = data.DataLoader(dataset=test_dataset,
                                    batch_size=HP['batch_size']*16,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=train_dataset.pad)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DittoModel(device=device,
                        lm=HP['lm'],
                        alpha_aug=HP['alpha_aug'])

    model.to(device)
    directory = os.path.join(HP['logdir'], HP['task'])
    ckpt_path = os.path.join(HP['logdir'], HP['task'], 'model.pt')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])



    all_probs = []
    with torch.no_grad():
        for  batch in test_iter:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            



    df = pd.DataFrame(all_probs, columns=['score'])
    df.to_csv('../SCORES/score_DITTO_'+task.split('/')[1]+'.csv', index=False)  

