import os
import numpy as np
import py_entitymatching as em

import pandas as pd



sens_dict = {
    'Walmart-Amazon':['category','printers'], # eq
    'Beer':['Beer_Name','Red'], # cont
    'Amazon-Google': ['manufacturer','Microsoft'], # cont
    'Fodors-Zagat':['type','asian'],  # eq
    'iTunes-Amazon':['Genre','Dance'], # cont
    'DBLP-GoogleScholar':['venue','vldb j'], # cont
    'DBLP-ACM':['authors','female'], # func
    'COMPAS': ['Ethnic_Code_Text', 'African-American']
}
datasets_dir_ = 'DATA/format2'
for task in [
    'Walmart-Amazon',
    'Beer',
    'Amazon-Google',
    'Fodors-Zagats',
    'iTunes-Amazon',
    'DBLP-GoogleScholar',
    'DBLP-ACM']:
        
    
    datasets_dir = datasets_dir_ + task


    for model in ['SVM', 'LogReg', 'LinReg']:
    
    
        path_A = datasets_dir + os.sep  + '/tableA.csv'
        path_B = datasets_dir + os.sep  + '/tableB.csv'
        path_train = datasets_dir + os.sep  + '/train.csv'
        path_test = datasets_dir + os.sep  + '/test.csv'

        A = em.read_csv_metadata(path_A, key='id')
        B = em.read_csv_metadata(path_B, key='id')
        try:
            tmp = pd.read_csv(path_train)
            tmp['id']
        except:
            tmp['id'] = range(len(tmp))
            path_train =  path_train.replace('.csv','__.csv')
            
            tmp.rename(columns={'ltable_id': 'left_id', 'rtable_id': 'right_id'}, inplace=True)


            tmp.to_csv(path_train , index = False)

            tmp = pd.read_csv(path_test)
            tmp['id'] = range(len(tmp))
            path_test =  path_test.replace('.csv','__.csv')
            tmp.rename(columns={'ltable_id': 'left_id', 'rtable_id': 'right_id'}, inplace=True)

            tmp.to_csv(path_test, index = False)






        I = em.read_csv_metadata(path_train, key='id',
                                ltable=A, rtable=B,
                                fk_ltable='left_id', fk_rtable='right_id')

        J = em.read_csv_metadata(path_test, key='id',
                                ltable=A, rtable=B,
                                fk_ltable='left_id', fk_rtable='right_id')

        # Create a set of ML-matchers
        if model == 'SVM':
            model_ = em.SVMMatcher(name='SVM', random_state=0, probability=True)
        elif model == 'LogReg':
            model_ = em.LogRegMatcher(name='LogReg', random_state=0, max_iter=1000)
        elif model == 'LinReg':
            model_ = em.LinRegMatcher(name='LinReg')



        atypes1 = em.get_attr_types(A)
        atypes2 = em.get_attr_types(B)

        block_c = em.get_attr_corres(A, B)
        tok = em.get_tokenizers_for_blocking()
        sim = em.get_sim_funs_for_blocking()
        F = em.get_features(A, B, atypes1, atypes2, block_c, tok, sim)


        H = em.extract_feature_vecs(I,
                                    feature_table=F,
                                    attrs_after='label',
                                    show_progress=True)

        H = em.impute_table(H, missing_val=np.nan)

        model_.fit(table=H,
                exclude_attrs=['id', 'left_id', 'right_id', 'label'],
                target_attr='label')

        L = em.extract_feature_vecs(J, feature_table=F,
                                    attrs_after='label', show_progress=True)

        L = em.impute_table(L, missing_val=np.nan)
        
        predictions = model_.predict(table=L, exclude_attrs=['id', 'left_id', 'right_id', 'label'],
                                    append=True, target_attr='preds', inplace=False, return_probs=True,
                                    probs_attr='proba')

        proba = np.array(predictions['proba'])
        
        label = L['label']
        col_left = []
        col_right = []
        df_right = pd.read_csv(path_A)
        df_left = pd.read_csv(path_A)
        for i,left in enumerate(list(L['left_id'])):
            right = list(L['right_id'])[i]

            left_ent = df_right[df_right['id'] ==right]
            right_ent = df_left[df_left['id'] ==left]

            if task == 'DBLP-GoogleScholar':
                key = 'journal'
            else:
                key = sens_dict[task][0] 
            
            t1 = str(list(left_ent[key])).replace('[','').replace(']','').replace('\'','').replace('`','').replace('\"','').strip()
            t2= str(list(right_ent[key])).replace('[','').replace(']','').replace('\'','').replace('`','').replace('\"','').strip()

            col_left.append(t1)
            col_right.append(t2)
            
        df = pd.DataFrame({'left':col_left, 'right':col_right, 'label': label, 'score':proba})


        df.to_csv('../SCORES/score_'+model+'_'+task+'.csv', index = False)
        print(model, task)

