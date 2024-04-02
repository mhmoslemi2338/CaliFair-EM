
import pandas as pd
import torch
from model.model import TranHGAT
from torch.utils import data
from model.dataset import Dataset
from train import initialize_and_train
import csv
import os






path_base = 'DATA'
for task in [
    'Walmart-Amazon',
    'Beer',
    'Amazon-Google',
    'Fodors-Zagats',
    'iTunes-Amazon',
    'DBLP-GoogleScholar',
    'DBLP-ACM']:
        

    data_dir = os.path.join(path_base, task)







    test_path = 'test.csv'
    train_path = 'train.csv'
    valid_path = 'valid.csv'




    run_id = 0
    batch_size = 16
    max_len = 1024
    lr = 1e-5
    n_epochs = 200
    finetuning = True
    save_model = True
    model_path = "saved_model_"+task+"/"
    lm_path = None
    split = True
    lm = 'bert'




    def dynamic_convert_csv_to_txt(input_csv, output_txt):
        with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_txt, mode='w', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            for row in reader:
                parts = {'left': '', 'right': ''}
                for col_name, value in row.items():
                    if '_' in col_name:
                        prefix, attribute = col_name.split('_', 1)
                        parts[prefix] += f"COL {attribute} VAL {value} "
                
                left_part = parts.get('left', '').strip()
                right_part = parts.get('right', '').strip()
                label = row.get('label', '').strip()
                
                outfile.write(f"{left_part} \t{right_part} \t{label}\n")


    for i in [test_path, valid_path, train_path]:
        dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('.csv','_.txt')))



    trainset = os.path.join(data_dir,train_path.replace('.csv','_.txt'))
    validset = os.path.join(data_dir,valid_path.replace('.csv','_.txt'))
    testset = os.path.join(data_dir,test_path.replace('.csv','_.txt'))


    # load train/dev/test sets
    train_dataset = Dataset(trainset, ["0", "1"], lm=lm, lm_path=lm_path, max_len=max_len, split=split)
    valid_dataset = Dataset(validset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)
    test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)



    args = {
        "batch_size": batch_size,
        "lr": lr,
        "n_epochs": n_epochs,
        "finetuning": finetuning,
        "save_model": save_model,
        "model_path": model_path,
        "lm_path": lm_path,
        "lm": lm}




    torch.cuda.empty_cache()

    initialize_and_train(train_dataset, valid_dataset, test_dataset, train_dataset.get_attr_num(), args, '1')

    torch.cuda.empty_cache()

    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    #----------------------------------------------------------------





    trainset = os.path.join(data_dir,train_path.replace('.csv','_.txt'))
    validset = os.path.join(data_dir,valid_path.replace('.csv','_.txt'))
    testset = os.path.join(data_dir,test_path.replace('.csv','_.txt'))


    # load train/dev/test sets
    train_dataset = Dataset(trainset, ["0", "1"], lm=lm, lm_path=lm_path, max_len=max_len, split=split)
    valid_dataset = Dataset(validset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)
    test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)




    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TranHGAT( train_dataset.get_attr_num(), device, finetuning, lm=lm, lm_path=lm_path)
    model.load_state_dict(torch.load(model_path+'/'+'model.pt', map_location= device))
    model.to(device)

    model.eval()
    all_probs = []
    test_iter = data.DataLoader(dataset=test_dataset, batch_size=batch_size*8,shuffle=False, num_workers=0, collate_fn=Dataset.pad)
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            _, x, y, _, masks = batch
            logits, y1, y_hat = model(x.to(device), y.to(device), masks.to(device))
            logits = logits.view(-1, logits.shape[-1])
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            


    df = pd.DataFrame(all_probs, columns=['score'])
    df.to_csv('SCORES/score_HierGAT_'+task+'.csv', index=False)  
    torch.cuda.empty_cache()
 


