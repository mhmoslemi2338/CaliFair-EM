import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import pandas as pd
from src.config import Config
from src.data_loader import load_data, DataType
from src.data_representation import DeepMatcherProcessor
from src.evaluation import Evaluation
from src.model import save_model
from src.optimizer import build_optimizer
from src.training import train
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_bert import BertForSequenceClassification







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


    num_epochs = 25
    model_type = 'bert' # bert
    model_name_or_path = 'bert-base-uncased' # "pre_trained_model/bert-base-uncased
    train_batch_size = 4
    eval_batch_size = 4
    max_seq_length = 256

    model_output_dir = 'MODEL_' + data_dir.split('/')[-1]
    weight_decay = 0
    max_grad_norm = 1
    warmup_steps = 0 
    adam_eps = 1e-8
    learning_rate = 2e-5
    save_model_after_epoch = False
    do_lower_case = True



    test_path = 'test.csv'
    train_path = 'train.csv'
    valid_path = 'valid.csv'
    for isx in [test_path, train_path, valid_path]:
        df = pd.read_csv(data_dir + '/'+ isx )
        df['combined_left'] = df.filter(like='left').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df['combined_right'] = df.filter(like='right').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df = df[['id' , 'combined_left','combined_right','label']]
        df = df.rename(columns={'id':'idx','combined_left': 'text_left', 'combined_right': 'text_right','label':'label'})
        df.to_csv(os.path.join(data_dir, isx.replace('csv','tsv')),sep='\t', index=False)



    # exp_name = create_experiment_folder(model_output_dir, model_type, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DeepMatcherProcessor()
    label_list = processor.get_labels()
    config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
    model = model_class.from_pretrained(model_name_or_path, config=config)
    model.to(device)







    train_examples = processor.get_train_examples(data_dir) 
    training_data_loader = load_data(train_examples,label_list,tokenizer,max_seq_length,train_batch_size,DataType.TRAINING, model_type)


    num_train_steps = len(training_data_loader) * num_epochs
    optimizer, scheduler = build_optimizer(model,num_train_steps,learning_rate,adam_eps,warmup_steps,weight_decay)
    eval_examples = processor.get_test_examples(data_dir)
    evaluation_data_loader = load_data(eval_examples,label_list,tokenizer,max_seq_length,eval_batch_size,DataType.EVALUATION, model_type)


    evaluation = Evaluation(evaluation_data_loader, model_output_dir, model_output_dir, len(label_list), model_type)




    train(device,
            training_data_loader,
            model,
            optimizer,
            scheduler,
            evaluation,
            num_epochs,
            max_grad_norm,
            save_model_after_epoch,
            experiment_name=model_output_dir,
            output_dir=model_output_dir,
            model_type=model_type)

    save_model(model, model_output_dir, model_output_dir, tokenizer=tokenizer)


    print('------- TRAINING DONE --------')
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

    try:
        torch.cuda.empty_cache()
        print('gpu cleaned')
    except:
        print('gpu clean failed')






    config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = tokenizer.from_pretrained(model_output_dir + '/' + model_output_dir, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)
    model = model.from_pretrained(model_output_dir + '/' + model_output_dir)


    model.to(device)



    test_batch_size = train_batch_size * 4
    processor = DeepMatcherProcessor()
    test_examples = processor.get_test_examples(data_dir)
    test_data_loader = load_data(test_examples,
                                    processor.get_labels(),
                                    tokenizer,
                                    max_seq_length,
                                    test_batch_size,
                                    DataType.TEST,model_type)



    labels = None
    all_probs = []
    all_y = [] 
    for batch in tqdm(test_data_loader, desc="Test"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device),
                        'attention_mask': batch[1].to(device),
                        'token_type_ids': batch[2].to(device),
                        'labels': batch[3].to(device)}

            outputs = model(**inputs)
            _, logits = outputs[:2]

            labels = inputs['labels'].detach().cpu().numpy()
            probs = logits.softmax(dim=1)[:, 1]

            all_probs += probs.cpu().numpy().tolist()
            all_y += labels.tolist()


    y_true = all_y
    y_score = all_probs
    df = pd.DataFrame({'prob': y_score, 'label': y_true})
    df.to_csv('SCORES/score_EMTransformer_'+data_dir.strip('/').split('/')[-1]+ '.csv', index=False)  
