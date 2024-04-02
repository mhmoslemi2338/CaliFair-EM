import pandas as pd
import os
import numpy as np

def prep_dataset(data_dir):
    data_path  = os.path.join('data',data_dir)

    test_path = 'test.csv'
    train_path = 'train.csv'
    valid_path = 'valid.csv'

    try:
        df_train = pd.read_csv(os.path.join(data_path,train_path.replace('.csv','_original.csv')))
        df_test = pd.read_csv(os.path.join(data_path,test_path.replace('.csv','_original.csv')))
        df_valid = pd.read_csv(os.path.join(data_path,valid_path.replace('.csv','_original.csv')))
        return
    except:
        df_train = pd.read_csv(os.path.join(data_path,train_path))
        df_test = pd.read_csv(os.path.join(data_path,test_path))
        df_valid = pd.read_csv(os.path.join(data_path,valid_path))



        df = pd.concat([df_train, df_test, df_valid], ignore_index=True)

        col = list(df.columns)

        left = {}
        right= {}
        for row in col:
            if 'left' in row:
                left[row] = row.replace('left','').replace('_','')
            if 'right' in row:
                right[row] = row.replace('right','').replace('_','')




        # selected_columns = ['left_title', 'left_authors', 'left_venue', 'left_year']
        df_left = df[list(left.keys())]
        df_right = df[list(right.keys())]

        df_right = df_right.rename(columns=right)
        df_left = df_left.rename(columns=left)


        tmp = {}
        for col in df_left:
            tmp[col] = df_left.drop_duplicates(subset  = col).shape[0]
        max_key = max(tmp, key=tmp.get)



        df_left = df_left.drop_duplicates(subset = max_key)
        df_right = df_right.drop_duplicates(subset = max_key)


        df_right['id'] = df_right.reset_index().index
        df_left['id'] = df_left.reset_index().index

        df_left = df_left.reset_index(drop=True)
        df_right = df_right.reset_index(drop=True)



        # r --> A
        # l --> B



        # df_right.to_csv(os.path.join(data_path, 'tableA.tsv'),sep='\t', index=False)
        # df_left.to_csv(os.path.join(data_path, 'tableB.tsv'),sep='\t', index=False)

        df_right.to_csv(os.path.join(data_path, 'tableA.csv'), index=False)
        df_left.to_csv(os.path.join(data_path, 'tableB.csv'), index=False)




        # pd.read_csv()
        data_path  = os.path.join('data',data_dir)

        test_path = 'test.csv'
        train_path = 'train.csv'
        valid_path = 'valid.csv'


        df_train = pd.read_csv(os.path.join(data_path,train_path))
        df_test = pd.read_csv(os.path.join(data_path,test_path))
        df_valid = pd.read_csv(os.path.join(data_path,valid_path))


        df_train.to_csv(os.path.join(data_path, 'train_original.csv'))
        df_test.to_csv(os.path.join(data_path, 'test_original.csv'))
        df_valid.to_csv(os.path.join(data_path, 'valid_original.csv'))



        df_test_ = pd.DataFrame(columns=['ltable_id', 'rtable_id', 'label'])
        df_train_ = pd.DataFrame(columns=['ltable_id', 'rtable_id', 'label'])
        df_valid_ = pd.DataFrame(columns=['ltable_id', 'rtable_id', 'label'])



        for index, row in df_train.iterrows():
            left_key = row['left_'+max_key]
            right_key = row['right_'+max_key]
            id_left = list(df_left[df_left[max_key] == left_key]['id'])[0]
            id_right = list(df_right[df_right[max_key] == right_key]['id'])[0]
            label = row['label']
            new_data = {'ltable_id': id_left, 'rtable_id': id_right, 'label': label}
            df_train_.loc[len(df_train_)] = new_data

        for index, row in df_test.iterrows():
            left_key = row['left_'+max_key]
            right_key = row['right_'+max_key]
            id_left = list(df_left[df_left[max_key] == left_key]['id'])[0]
            id_right = list(df_right[df_right[max_key] == right_key]['id'])[0]
            label = row['label']
            new_data = {'ltable_id': id_left, 'rtable_id': id_right, 'label': label}
            df_test_.loc[len(df_test_)] = new_data


        for index, row in df_valid.iterrows():
            left_key = row['left_'+max_key]
            right_key = row['right_'+max_key]
            id_left = list(df_left[df_left[max_key] == left_key]['id'])[0]
            id_right = list(df_right[df_right[max_key] == right_key]['id'])[0]
            label = row['label']
            new_data = {'ltable_id': id_left, 'rtable_id': id_right, 'label': label}
            df_valid_.loc[len(df_valid_)] = new_data



        # df_train_.to_csv(os.path.join(data_path, 'train.tsv'),sep='\t', index=False)
        # df_test_.to_csv(os.path.join(data_path, 'test.tsv'),sep='\t', index=False)
        # df_valid_.to_csv(os.path.join(data_path, 'valid.tsv'),sep='\t', index=False)


        df_train_.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        df_test_.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        df_valid_.to_csv(os.path.join(data_path, 'valid.csv'), index=False)

