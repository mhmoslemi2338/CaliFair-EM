import gender_guesser.detector as gender
import copy
import ot
import pandas as pd

import numpy as np
female_names = ['adriana', 'agma', 'alexandra', 'alice', 'aya', 'barbara', 'betty', 'bhavani', 'carol',
                    'carole', 'cass', 'cecilia', 'chia-jung', 'christine', 'clara', 'claudia', 'debra',
                    'diane', 'dimitra', 'ebru', 'elaheh', 'elena', 'elisa', 'elke', 'esther', 'fatima',
                    'fatma', 'felicity', 'françoise', 'gillian', 'hedieh', 'helen', 'ilaria', 'isabel', 
                    'jeanette', 'jeanine', 'jennifer', 'jenny', 'joann', 'julia', 'juliana', 'julie', 
                    'kelly', 'kimberly', 'laura', 'letizia', 'ljiljana', 'louiqa', 'lynn', 'maria',
                    'marianne', 'melissa', 'meral', 'monica', 'myra', 'pamela', 'patricia', 'paula',
                    'pierangela', 'pina', 'rachel', 'sandra', 'sheila', 'sihem', 'silvana', 'sophie',
                    'sorana', 'sunita', 'susan', 'teresa', 'tova', 'ulrike', 'vana', 'véronique', 'ya-hui', 'yelena', 'zoé']

def gender_rev(name):
    name = name.strip().split()[0].strip()
    d = gender.Detector()
    modified_name = name[0].upper() + name[1:]
    gen_dict = {'male':'male',
    'female':'female',
    'andy':'other',
    'mostly_male': 'male',
    'mostly_female': 'female',
    'unknown': 'other'}
    return gen_dict[d.get_gender(modified_name)]


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


def make_sens_vector(df , dataset, sens_dict):

    if dataset in ['Walmart-Amazon','Fodors-Zagat', 'COMPAS']:
        df['left_contains_s'] = df['left_'+sens_dict[dataset][0]] == sens_dict[dataset][1]
        df['right_contains_s'] = df['right_'+sens_dict[dataset][0]] == sens_dict[dataset][1]

    elif dataset in ['Beer', 'Amazon-Google',  'iTunes-Amazon', 'DBLP-GoogleScholar']:
        df['left_'+sens_dict[dataset][0]] =  df['left_'+sens_dict[dataset][0]].astype(str)
        df['right_'+sens_dict[dataset][0]] =  df['right_'+sens_dict[dataset][0]].astype(str)
        df['left_contains_s'] = df['left_'+sens_dict[dataset][0]].str.lower().str.contains(sens_dict[dataset][1].lower())
        df['right_contains_s'] = df['right_'+sens_dict[dataset][0]].str.lower().str.contains(sens_dict[dataset][1].lower())
    else:
        df['left_'+sens_dict[dataset][0]] =  df['left_'+sens_dict[dataset][0]].astype(str)
        df['right_'+sens_dict[dataset][0]] =  df['right_'+sens_dict[dataset][0]].astype(str)

        df['left_contains_s'] = df['left_'+sens_dict[dataset][0]].apply(lambda x: x.replace('&#216;','').replace('&#214;','').replace('&#237;',',').split(',')[-1].strip())
        df['right_contains_s'] = df['right_'+sens_dict[dataset][0]].apply(lambda x: x.replace('&#216;','').replace('&#214;','').replace('&#237;',',').split(',')[-1].strip())
        
        df['left_contains_s'] = df['left_contains_s'].apply(lambda x: ', '.join([gender_rev(name) for name in x.split(',')]))
        df['right_contains_s'] = df['right_contains_s'].apply(lambda x: ', '.join([gender_rev(name) for name in x.split(',')]))

        df['left_contains_s'] = df['left_contains_s'].apply(lambda x: 'True' if 'female' in str(x) else 'False')
        df['right_contains_s'] = df['right_contains_s'].apply(lambda x: 'True' if 'female' in str(x) else 'False')
        
        df['left_contains_s'] = df['left_contains_s'].apply(lambda x: any(item in x for item in ['True']))
        df['right_contains_s'] = df['right_contains_s'].apply(lambda x: any(item in x for item in ['True']))

        # df['left_contains_s'] = df['left_contains_s'].apply(lambda x: any(item in x for item in female_names))
        # df['right_contains_s'] = df['right_contains_s'].apply(lambda x: any(item in x for item in female_names))

    result_vector = np.logical_or(df['left_contains_s'], df['right_contains_s']).astype(int)
    sens_attr = np.array(result_vector).reshape(-1)

    return sens_attr




############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################



def calc_DP_TPR(sens_attr,y_true,y_score ):
    nums = 1000
    Distributional_Parity_TPR = 0
    groups = list(np.unique(sens_attr))
    for i,g1 in enumerate(groups):
        for g2 in groups[i+1:]:

            y_g1 = y_true[sens_attr == g1 ]
            S_g1 = y_score[sens_attr == g1 ]

            y_g2 = y_true[sens_attr == g2 ]
            S_g2 = y_score[sens_attr == g2 ]

            S_g1_TPR = S_g1[y_g1 ==1]
            S_g2_TPR = S_g2[y_g2 ==1]

            expected_val = 0
            for thresh in np.linspace(0, 1, nums):


                P1 = np.sum((S_g1_TPR > thresh)) / len(S_g1_TPR)
                P2 = np.sum((S_g2_TPR > thresh)) / len(S_g2_TPR)

                expected_val += np.abs(P1 - P2)

            expected_val = expected_val / nums
            Distributional_Parity_TPR += expected_val
    return Distributional_Parity_TPR
        


def calc_DP_FPR(sens_attr,y_true,y_score ):
    nums = 1000
    Distributional_Parity_FPR = 0
    groups = list(np.unique(sens_attr))

    for i,g1 in enumerate(groups):
        for g2 in groups[i+1:]:

            y_g1 = y_true[sens_attr == g1 ]
            S_g1 = y_score[sens_attr == g1 ]

            y_g2 = y_true[sens_attr == g2 ]
            S_g2 = y_score[sens_attr == g2 ]

            S_g1_FPR = S_g1[y_g1 ==0]
            S_g2_FPR = S_g2[y_g2 ==0]

            expected_val = 0
            for thresh in np.linspace(0, 1, nums):


                P1 = np.sum((S_g1_FPR > thresh)) / len(S_g1_FPR)
                P2 = np.sum((S_g2_FPR > thresh)) / len(S_g2_FPR)

                expected_val += np.abs(P1 - P2)

            expected_val = expected_val / nums
            Distributional_Parity_FPR += expected_val
    return Distributional_Parity_FPR
        




def calc_SDD(sens_attr,y_true,y_score ):
    nums = 1000
    SDD = 0
    groups = list(np.unique(sens_attr))
    for i,g1 in enumerate(groups):

        y_g1 = y_true[sens_attr == g1 ]
        S_g1 = y_score[sens_attr == g1 ]

        expected_val_BG = 0
        for thresh in np.linspace(0, 1, nums):

            P1 = np.sum((S_g1 > thresh)) / len(S_g1)
            P_BG = np.sum((y_score > thresh)) / len(y_score)
            expected_val_BG += np.abs(P1 - P_BG)

        expected_val_BG = expected_val_BG / nums

        SDD += expected_val_BG
    return SDD




def calc_SPDD(sens_attr,y_true,y_score ):
    nums = 1000
    SPDD = 0
    groups = list(np.unique(sens_attr))
    for i,g1 in enumerate(groups):
        for g2 in groups[i+1:]:

            y_g1 = y_true[sens_attr == g1 ]
            S_g1 = y_score[sens_attr == g1 ]

            y_g2 = y_true[sens_attr == g2 ]
            S_g2 = y_score[sens_attr == g2 ]
            
            expected_val = 0
            for thresh in np.linspace(0, 1, nums):


                P1 = np.sum((S_g1 > thresh)) / len(S_g1)
                P2 = np.sum((S_g2 > thresh)) / len(S_g2)

                expected_val += np.abs(P1 - P2)

            expected_val = expected_val / nums
            SPDD += expected_val
    return SPDD

# FNR + FPR





def calc_DP_PR(sens_attr,y_true,y_score ):
    nums = 1000
    PR = 0


    S_g1 = y_score[sens_attr == 1 ]
    S_g2 = y_score[sens_attr == 0 ]
    
    expected_val = 0
    for thresh in np.linspace(0, 1, nums):


        P1 = np.sum((S_g1 > thresh)) / len(S_g1)
        P2 = np.sum((S_g2 > thresh)) / len(S_g2)

        expected_val += np.abs(P1 - P2)

    expected_val = expected_val / nums
    return expected_val



def calc_EO_disp(sens_attr,y_true,y_score ):
    nums = 1000
    Distributional_Parity_FPR = 0
    groups = list(np.unique(sens_attr))



    y_g1 = y_true[sens_attr == 1 ]
    S_g1 = y_score[sens_attr == 1 ]

    y_g2 = y_true[sens_attr == 0 ]
    S_g2 = y_score[sens_attr == 0 ]

    


    # S_g1_FPR = S_g1[y_g1 ==0]
    # S_g2_FPR = S_g2[y_g2 ==0]

    # S_g1_FNR = S_g1[y_g1 ==1]
    # S_g2_FNR = S_g2[y_g2 ==1]



    expected_val = 0
    for thresh in np.linspace(0, 1, nums):


        # P1 = np.sum((S_g1_FPR > thresh)) / len(S_g1_FPR)
        # P2 = np.sum((S_g2_FPR > thresh)) / len(S_g2_FPR)

        # P11 = np.sum((S_g1_FNR < thresh)) / len(S_g1_FNR)
        # P22 = np.sum((S_g2_FNR < thresh)) / len(S_g2_FNR)

        

        # expected_val += np.abs((P11 + P1) - (P22 + P2))


        FP_g1 = np.sum((S_g1 > thresh)[y_g1 ==0])
        FP_g2 = np.sum((S_g2 > thresh)[y_g2==0])

        TN_g1 =  np.sum((S_g1 <= thresh)[y_g1 ==0])
        TN_g2 =  np.sum((S_g2 <= thresh)[y_g2 ==0])

        fpr_g1 = FP_g1 / (FP_g1 + TN_g1)
        fpr_g2 = FP_g2 / (FP_g2 + TN_g2)

        # fpr = fp / (fp + tn) # False Positive Rate

        # fnr = fn / (fn + tp) # False Negative Rate

        FN_g1 =  np.sum((S_g1 <= thresh)[y_g1 ==1])
        TP_g1 = np.sum((S_g1 > thresh)[y_g1 ==1])

        FN_g2 =  np.sum((S_g2 <= thresh)[y_g2 ==1])
        TP_g2 = np.sum((S_g2 > thresh)[y_g2 ==1])

        # fnr_g1 = FN_g1 / (FN_g1 + TP_g1)
        # fnr_g2 = FN_g2 / (FN_g2 + TP_g2)

        tpr_g1 = TP_g1 / (FN_g1 + TP_g1)
        tpr_g2 = TP_g2 / (FN_g2 + TP_g2)


                # tpr + fpr

        expected_val+= np.abs((tpr_g1 + fpr_g1) - (tpr_g2 + fpr_g2))


    expected_val = expected_val / nums
    Distributional_Parity_FPR += expected_val
    return Distributional_Parity_FPR
        


import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_additional_fairness_metrics(y_true, y_pred, sensitive_att):
    # Initialize dictionaries to hold metrics for sensitive and non-sensitive groups
    metrics_sensitive = {}
    metrics_nonsensitive = {}
    
    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) # True Positive Rate
        fpr = fp / (fp + tn) # False Positive Rate
        fnr = fn / (fn + tp) # False Negative Rate
        tnr = tn / (tn + fp) # True Negative Rate
        ppv = tp / (tp + fp) # Positive Predictive Value
        npv = tn / (tn + fn) # Negative Predictive Value
        fdr = fp / (fp + tp) # False Discovery Rate
        for_ = fn / (fn + tn) # False Omission Rate
        return {'TPR': tpr, 'FPR': fpr, 'FNR': fnr, 'TNR': tnr, 'PPV': ppv, 'NPV': npv, 'FDR': fdr, 'FOR': for_}
    

    # Separate data into sensitive and non-sensitive groups
    indices_sensitive = np.where(sensitive_att == 1)
    indices_nonsensitive = np.where(sensitive_att == 0)
    
    # Calculate metrics for sensitive group
    metrics_sensitive = calculate_metrics(y_true[indices_sensitive], y_pred[indices_sensitive])
    e_odds_sens = metrics_sensitive['TPR'] +metrics_sensitive['FPR']
    e_opp__sens = metrics_sensitive['TPR']
    # Calculate metrics for non-sensitive group
    metrics_nonsensitive = calculate_metrics(y_true[indices_nonsensitive], y_pred[indices_nonsensitive])
    e_odds__non_sens = metrics_nonsensitive['TPR'] +metrics_nonsensitive['FPR']
    e_opp__non_sens = metrics_nonsensitive['TPR']
    # Calculate parity differences
    parity_differences = {metric: metrics_sensitive[metric] - metrics_nonsensitive[metric] for metric in metrics_sensitive}
    modified_dict = {key + ' partiy': value for key, value in parity_differences.items()}



    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_att = np.array(sensitive_att)
    y_true_sensitive = y_true[sensitive_att == 1]
    y_pred_sensitive = y_pred[sensitive_att == 1]
    y_true_nonsensitive = y_true[sensitive_att == 0]
    y_pred_nonsensitive = y_pred[sensitive_att == 0]
    sp = y_pred_sensitive.mean() - y_pred_nonsensitive.mean()
    accuracy_sensitive = (y_pred_sensitive == y_true_sensitive).mean()
    accuracy_nonsensitive = (y_pred_nonsensitive == y_true_nonsensitive).mean()
    accuracy_parity = accuracy_sensitive - accuracy_nonsensitive
    
    modified_dict['Statistical Parity'] = sp
    modified_dict['Accuracy Parity'] = accuracy_parity

    modified_dict['e_odds_sens'] = e_odds_sens
    modified_dict['e_odds__non_sens'] = e_odds__non_sens


    modified_dict['e_opp__sens'] = e_opp__sens
    modified_dict['e_opp__non_sens'] = e_opp__non_sens



    return modified_dict, 


def calculate_additional_fairness_metrics2(y_true, y_pred, sensitive_att):
    # Initialize dictionaries to hold metrics for sensitive and non-sensitive groups
    metrics_sensitive = {}
    metrics_nonsensitive = {}
    
    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) # True Positive Rate
        fpr = fp / (fp + tn) # False Positive Rate
        fnr = fn / (fn + tp) # False Negative Rate
        tnr = tn / (tn + fp) # True Negative Rate

        return {'TPR': tpr, 'FPR': fpr, 'FNR': fnr, 'TNR': tnr}
        
        
        # (tp + tn) / (tp+tn+fp+fn)
        
        # tp / (tp + fn)
    


    
    # Separate data into sensitive and non-sensitive groups
    indices_sensitive = np.where(sensitive_att == 1)
    indices_nonsensitive = np.where(sensitive_att == 0)
    
    # Calculate metrics for sensitive group
    metrics_sensitive = calculate_metrics(y_true[indices_sensitive], y_pred[indices_sensitive])
    e_odds_sens = metrics_sensitive['TPR'] +metrics_sensitive['FPR']
    e_opp__sens = metrics_sensitive['TPR']

    # Calculate metrics for non-sensitive group
    metrics_nonsensitive = calculate_metrics(y_true[indices_nonsensitive], y_pred[indices_nonsensitive])
    e_odds__non_sens = metrics_nonsensitive['TPR'] +metrics_nonsensitive['FPR']
    e_opp__non_sens = metrics_nonsensitive['TPR']
    # Calculate parity differences
    parity_differences = {metric: metrics_sensitive[metric] - metrics_nonsensitive[metric] for metric in metrics_sensitive}
    modified_dict = {key + ' partiy': value for key, value in parity_differences.items()}



    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_att = np.array(sensitive_att)
    y_true_sensitive = y_true[sensitive_att == 1]
    y_pred_sensitive = y_pred[sensitive_att == 1]
    y_true_nonsensitive = y_true[sensitive_att == 0]
    y_pred_nonsensitive = y_pred[sensitive_att == 0]
    sp = y_pred_sensitive.mean() - y_pred_nonsensitive.mean()
    accuracy_sensitive = (y_pred_sensitive == y_true_sensitive).mean()
    accuracy_nonsensitive = (y_pred_nonsensitive == y_true_nonsensitive).mean()
    accuracy_parity = accuracy_sensitive - accuracy_nonsensitive
    
    modified_dict['Statistical Parity'] = sp
    modified_dict['Accuracy Parity'] = accuracy_parity

    modified_dict['e_odds_sens'] = e_odds_sens
    modified_dict['e_odds__non_sens'] = e_odds__non_sens


    modified_dict['e_opp__sens'] = e_opp__sens
    modified_dict['e_opp__non_sens'] = e_opp__non_sens



    return modified_dict, 




########################################################################
########################################################################
########################################################################
########################################################################
########################################################################


def calc_bin(data):
    try:
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        bin_width = 2 * IQR / (len(data) ** (1/3))
        data_range = np.max(data) - np.min(data)
        num_bins = int(np.round(data_range / bin_width))
    except:
        num_bins = 20

    return num_bins


def calc_bary(score,sens_attr, R = False):
    score_g1 = score[sens_attr == 1]
    score_g2 = score[sens_attr == 0]



    num  = min(int(max(calc_bin(score_g1)*1.5, calc_bin(score_g2)*1.5)), 500)
    hist1, bin_edges1 = np.histogram(score_g1, bins=np.linspace(0, 1, num+1 ))
    hist2, bin_edges2 = np.histogram(score_g2, bins=np.linspace(0, 1, num+1 ))


    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])


    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    hist1[hist1 == 0 ] = 1e-20
    hist2[hist2 == 0 ] = 1e-20

    a1 = hist1
    a2 = hist2


    M  =ot.utils.dist(hist1.reshape(-1,1),metric='cityblock')
    M /= M.max()
    A = np.vstack((hist1, hist2)).T

    weight = 0.5 # 0<=weight<=1
    weights = np.array([1 - weight, weight])

    # wasserstein
    reg = 1e-10
    alpha = 1
    bary_wass = ot.unbalanced.barycenter_unbalanced(A, M, reg, alpha, weights=weights)

    if R:
        return bary_wass, A, bin_centers1, bin_centers2

    return bary_wass




def map_scores(bary_wass,score, sens_attr ):

    score_g1 = score[sens_attr == 1]
    score_g2 = score[sens_attr == 0]

    mapper1 = ot.da.MappingTransport(mu=.001, eta=1e-8, bias=False, max_iter=300, verbose= True, kernel = 'gaussian', sigma = 2)
    mapper1.fit(Xs=score_g1.reshape(-1, 1),Xt = bary_wass.reshape(-1, 1))

    mapper2 = ot.da.MappingTransport(mu=.001, eta=1e-8, bias=False, max_iter=300, verbose= True, kernel = 'gaussian', sigma = 2)
    mapper2.fit(Xs=score_g2.reshape(-1, 1), Xt = bary_wass.reshape(-1, 1))

    # Use the mapper to transform score lists to the distribution of the barycenter
    scores_list_1_mapped = mapper1.transform(Xs=score_g1.reshape(-1, 1)).ravel()
    scores_list_2_mapped = mapper2.transform(Xs=score_g2.reshape(-1, 1)).ravel()



    map_score = np.zeros(score.shape)
    map_score[sens_attr == 1] = scores_list_1_mapped
    map_score[sens_attr == 0] = scores_list_2_mapped

    return map_score



def PR_make(PR_total, PR_g1, PR_g2,y_true, y_pred ,sens_attr):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    PR_total.append((tp + fp) / (tp+tn+fp+fn))

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1.append((tp + fp) / (tp+tn+fp+fn))

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2.append((tp + fp) / (tp+tn+fp+fn))
    return PR_total, PR_g1, PR_g2


from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score


def ACC_make(ACC_total, ACC_g1, ACC_g2,y_true, y_pred ,sens_attr):

    accuracy = accuracy_score(y_true, y_pred)
    ACC_total.append(accuracy)

    accuracy = accuracy_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    ACC_g1.append(accuracy)

    accuracy = accuracy_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    ACC_g2.append(accuracy)
    return ACC_total, ACC_g1, ACC_g2



def F1_make(F1_total, F1_g1, F1_g2,y_true, y_pred ,sens_attr):

    accuracy = f1_score(y_true, y_pred)
    F1_total.append(accuracy)

    accuracy = f1_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    F1_g1.append(accuracy)

    accuracy = f1_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    F1_g2.append(accuracy)
    return F1_total, F1_g1, F1_g2

from sklearn.metrics import roc_curve, roc_auc_score


def E_make(E_op_g1, E_op_g2,E_od_g1, E_od_g2, y_true,y_pred,  sens_attr):
    additional_fairness_metrics = calculate_additional_fairness_metrics2(y_true, y_pred, sens_attr)[0]
    E_op_g1.append(additional_fairness_metrics['e_opp__sens'])
    E_op_g2.append(additional_fairness_metrics['e_opp__non_sens'] )
    E_od_g1.append(additional_fairness_metrics['e_odds_sens'])
    E_od_g2.append(additional_fairness_metrics['e_odds__non_sens'])
    return E_op_g1, E_op_g2,E_od_g1, E_od_g2



def AUC_make(y_true, y_score, sens_attr):
    auc = roc_curve(y_true, y_score)
    auc_sens = roc_curve(y_true[sens_attr ==1], y_score[sens_attr ==1])
    auc_non_snes = roc_curve(y_true[sens_attr ==0], y_score[sens_attr ==0])
    return auc, auc_sens, auc_non_snes





def calc_metric_plt(score_in, y_true,sens_attr):

    y_score = score_in
    AUC_tot , AUC_g1, AUC_g2 = AUC_make(y_true, y_score, sens_attr)


    EO_opps_distri = calc_DP_TPR(sens_attr, y_true, y_score)
    EO_odds_distri = calc_EO_disp(sens_attr, y_true, y_score)

    PR_total, PR_g1, PR_g2 = [], [], [] 
    ACC_total, ACC_g1, ACC_g2 = [], [], [] 
    F1_total, F1_g1, F1_g2 = [], [], [] 

    E_op_g1,E_op_g2 =[], []
    E_od_g1, E_od_g2 =[], []
    range = np.linspace(0, 1, 100)
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        E_op_g1, E_op_g2,E_od_g1, E_od_g2 = E_make(E_op_g1, E_op_g2,E_od_g1, E_od_g2, y_true,y_pred,  sens_attr)
        PR_total, PR_g1, PR_g2 = PR_make(PR_total, PR_g1, PR_g2,y_true, y_pred , sens_attr)
        ACC_total, ACC_g1, ACC_g2 = ACC_make(ACC_total, ACC_g1, ACC_g2,y_true, y_pred ,sens_attr)
        F1_total, F1_g1, F1_g2 = F1_make(F1_total, F1_g1, F1_g2,y_true, y_pred ,sens_attr)

    METRICS = [AUC_tot,AUC_g1, AUC_g2, E_op_g1, E_op_g2,E_od_g1,E_od_g2,PR_total, PR_g1, PR_g2,
                  ACC_total, ACC_g1, ACC_g2, F1_total, F1_g1, F1_g2, EO_opps_distri, EO_odds_distri]
    
    METRICS = {
        'AUC_tot': AUC_tot,
        'AUC_g1': AUC_g1,
        'AUC_g2': AUC_g2,
        'E_op_g1': E_op_g1,
        'E_op_g2': E_op_g2,
        'E_od_g1': E_od_g1,
        'E_od_g2': E_od_g2,
        'PR_total': PR_total,
        'PR_g1': PR_g1,
        'PR_g2': PR_g2,
        'ACC_total': ACC_total,
        'ACC_g1': ACC_g1,
        'ACC_g2': ACC_g2,
        'F1_total': F1_total,
        'F1_g1': F1_g1,
        'F1_g2': F1_g2,
        'EO_opps_distri': EO_opps_distri,
        'EO_odds_distri': EO_odds_distri
    }

    return METRICS




import matplotlib.pyplot as plt
import os
def my_plt(G_dict,color_dict, KEYS,METRICS, F, F_title, L , size , F_legend, DATASET, MODEL, stage):
    if not os.path.exists('Paper_figures'):
        os.makedirs('Paper_figures')
    if not os.path.exists('Paper_figures/'+stage):
        os.makedirs('Paper_figures/'+stage)
    metric_pre = []
    for k in KEYS:
        VAL = METRICS[k]
        if k in ['EO_opps_distri','EO_odds_distri']: continue
        

        if 'ACC' in k:metric = 'ACC'
        elif 'AUC' in k:metric = 'AUC'
        elif 'E_op' in k:metric = 'Equalized opportunity'
        elif 'E_od' in k:metric = 'Equalized odds'
        elif 'F1' in k:metric = 'F1-score'
        elif 'PR' in k: metric = 'Positive Rate'
        else: metric = ''

        if 'g1' in k:group = 'minority'
        elif 'g2' in k:group = 'majority'
        elif 'tot' in k:group = 'total'
        else:group =''

        if group not in G_dict[metric]: continue
        if metric not in metric_pre:
            metric_pre.append(metric)
            G_flag = [group]
            plt.figure(figsize=size)
            plt.xticks(fontsize = F)
            plt.yticks(fontsize = F)
            if metric != 'AUC':
                if metric == 'Equalized opportunity':  plt.ylabel('EO', fontsize = F_title)
                elif metric == 'Equalized odds':  plt.ylabel('EOD', fontsize = F_title)
                elif metric == 'F1-score': plt.ylabel('F1', fontsize = F_title)
                else: plt.ylabel(metric, fontsize = F_title)
                plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
            else:
                plt.xlabel('FPR', fontsize =F_title)  
                plt.ylabel('TPR', fontsize =F_title)        
        else: G_flag.append(group)

        
        if metric == 'AUC':
            x = VAL[0]
            y = VAL[1]
        else:
            x= np.linspace(0, 1, 100)
            y = VAL
            
        plt.plot(x,y,label =group, color = color_dict[group], linewidth = L)
        if sorted(G_flag) == sorted(G_dict[metric]):
            if metric!='Equalized odds':
                plt.ylim([0,1.03])
            plt.xlim([-0.02,1])
            legend =plt.legend(fontsize = F_legend)
            legend.get_frame().set_edgecolor('black')
            plt.gca().get_xticklabels()[0].set_horizontalalignment('right')
            plt.gca().get_yticklabels()[0].set_verticalalignment('center')
            plt.tight_layout()
            plt.savefig('Paper_figures/'+stage+'/'+DATASET+'_'+MODEL+'_'+metric+'_'+stage+'.pdf')
            plt.close()





def calc_bary2(score,sens_attr, R = False):
    score_g1 = score[sens_attr == 1]
    score_g2 = score[sens_attr == 0]



    num  = min(int(max(calc_bin(score_g1), calc_bin(score_g2))), 400)
    
    hist1, bin_edges1 = np.histogram(score_g1, bins=np.linspace(0, 1, num+1 ))
    hist2, bin_edges2 = np.histogram(score_g2, bins=np.linspace(0, 1, num+1 ))


    bin_centers1 = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])


    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # hist1[hist1 == 0 ] = 1e-20
    # hist2[hist2 == 0 ] = 1e-20




    M  =ot.dist(hist2.reshape(-1,1,),hist2.reshape(-1,1),metric='cityblock')
    # M  =ot.utils.dist0(hist1.shape[0])
    M /= M.max()
    A = np.vstack((hist1, hist2)).T

    weight = 0.5 # 0<=weight<=1
    weights = np.array([1 - weight, weight])

    # wasserstein
    reg = 8e-4
    alpha = 1e-4
    # bary_wass = ot.unbalanced.barycenter_unbalanced(A, M, reg, alpha, weights=weights)
    # bary_wass = ot.barycenter(A, M, reg, weights=weights, method='sinkhorn', numItermax = 10000000, stopThr= 1e-12)
    # bary_wass = ot.bregman.barycenter(A, M, reg, weights=weights, method='sinkhorn', numItermax = 10000000, stopThr= 1e-12)
    bary_wass=  ot.lp.barycenter(A,M, weights=weights, solver='interior-point')

    if R:
        return bary_wass, A, bin_centers1, bin_centers2

    return bary_wass





################################################################
################################################################
################################################################
################################################################
################################################################



def do_job(df , score,sens_attr, y_true, dataset, model, G_dict, color_dict, F, F_title, L , size , F_legend, stage,gamma):
    if not os.path.exists('Paper_figures'):
        os.makedirs('Paper_figures')

    auc_g1 = roc_auc_score(y_true[sens_attr==1], score[sens_attr ==1])
    auc_g2 = roc_auc_score(y_true[sens_attr==0], score[sens_attr ==0])
    auc = roc_auc_score(y_true, score)
    
    

    Eodd_disp = calc_EO_disp(sens_attr, y_true, score)
    Eop_disp = calc_DP_TPR(sens_attr, y_true, score)
    PR_disp = calc_DP_PR(sens_attr, y_true, score)

    y_pred = np.array([1 if score > 0.5 else 0 for score in score])

    f1_g1_5 = f1_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    f1_g2_5 = f1_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    f1_5 = f1_score(y_true, y_pred)
    
    accuracy_g1_5 = accuracy_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    accuracy_g2_5 = accuracy_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    accuracy_5 = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1_5 = (tp + fp) / (tp+tn+fp+fn)
    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2_5 = (tp + fp) / (tp+tn+fp+fn)

    additional_fairness_metrics = calculate_additional_fairness_metrics2(y_true, y_pred, sens_attr)[0]
    E_op_g1 = (additional_fairness_metrics['e_opp__sens'])
    E_op_g2 = (additional_fairness_metrics['e_opp__non_sens'] )
    E_od_g1 = (additional_fairness_metrics['e_odds_sens'])
    E_od_g2 = (additional_fairness_metrics['e_odds__non_sens'])


    E_op_5 = np.abs(E_op_g1 - E_op_g2)
    E_od_5 = np.abs(E_od_g2 - E_od_g1)
    PR_5 = np.abs(PR_g1_5 - PR_g2_5)
    delta_auc = np.abs(auc_g1 - auc_g2)


################################

    y_pred = np.array([1 if score > 0.9 else 0 for score in score])

    f1_g1_9 = f1_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    f1_g2_9 = f1_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    f1_9 = f1_score(y_true, y_pred)

    accuracy_g1_9 = accuracy_score(y_true[sens_attr ==1], y_pred[sens_attr ==1])
    accuracy_g2_9 = accuracy_score(y_true[sens_attr ==0], y_pred[sens_attr ==0])
    accuracy_9 = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==1], y_pred[sens_attr ==1]).ravel()
    PR_g1_9 = (tp + fp) / (tp+tn+fp+fn)
    tn, fp, fn, tp = confusion_matrix(y_true[sens_attr ==0], y_pred[sens_attr ==0]).ravel()
    PR_g2_9 = (tp + fp) / (tp+tn+fp+fn)

    additional_fairness_metrics = calculate_additional_fairness_metrics2(y_true, y_pred, sens_attr)[0]
    E_op_g1 = (additional_fairness_metrics['e_opp__sens'])
    E_op_g2 = (additional_fairness_metrics['e_opp__non_sens'] )
    E_od_g1 = (additional_fairness_metrics['e_odds_sens'])
    E_od_g2 = (additional_fairness_metrics['e_odds__non_sens'])



    E_op_9 = np.abs(E_op_g1 - E_op_g2)
    E_od_9 = np.abs(E_od_g2 - E_od_g1)
    PR_9 = np.abs(PR_g1_9 - PR_g2_9)











    METRICS_dict = {
    'Dataset': dataset,
    'Model': model,
    'optimal lambda': gamma,
    'Distributioanl disparity: Equal opportunity (TPR)': Eop_disp ,
    'Distributioanl disparity: Equalized odds': Eodd_disp,
    'Distributioanl disparity: PR': PR_disp,

    'Positive Rate Parity (Threshold = 0.5)': PR_5,
    'Equalized odds Parity (Threshold = 0.5)': E_od_5,
    'Equal opportunity Parity (Threshold = 0.5)': E_op_5,
    
    'Total Accuracy (Threshold = 0.5)': accuracy_5,
    'Minority Accuracy (Threshold = 0.5)': accuracy_g1_5,
    'Majority Accuracy (Threshold = 0.5)': accuracy_g2_5,

    'Total F1 (Threshold = 0.5)': f1_5,
    'Minority F1 (Threshold = 0.5)': f1_g1_5,
    'Majority F1 (Threshold = 0.5)': f1_g2_5,


    'Positive Rate Parity (Threshold = 0.9)': PR_9,
    'Equalized odds Parity (Threshold = 0.9)': E_od_9,
    'Equal opportunity Parity (Threshold = 0.9)': E_op_9,
    
    'Total Accuracy (Threshold = 0.9)': accuracy_9,
    'Minority Accuracy (Threshold = 0.9)': accuracy_g1_9,
    'Majority Accuracy (Threshold = 0.9)': accuracy_g2_9,

    'Total F1 (Threshold = 0.9)': f1_9,
    'Minority F1 (Threshold = 0.9)': f1_g1_9,
    'Majority F1 (Threshold = 0.9)': f1_g2_9,


    'Total AUC': auc,
    'Minority AUC': auc_g1,
    'Majority AUC': auc_g2,
    'Delta AUC': np.abs(auc_g1 - auc_g2),
    


    }


    df_new = pd.DataFrame(METRICS_dict, index=[0])


    try:
        df = pd.concat([df, df_new], ignore_index=True)
    except:
        df = copy.deepcopy(df_new)

    # if dataset =='Amazon-Google' and model =='HierMatcher':
    # METRICS = calc_metric_plt(score, y_true,sens_attr)
    # my_plt(G_dict,color_dict, list(METRICS.keys()),METRICS, F, F_title, L , size , F_legend, dataset, model, stage)





    return df










def plot_bef_after(score_optimal_Eop,score_optimal_Eodd,score_optimal_PR, model,dataset,sens_attr, y_true, score):


    y_score = score



    Eodd_disp_init = calc_EO_disp(sens_attr, y_true, y_score)
    auc_init = roc_auc_score(y_true, y_score)
    PR_disp_init = calc_DP_PR(sens_attr, y_true, y_score)
    Eop_disp_init = calc_DP_TPR(sens_attr, y_true, y_score)


    range = np.linspace(0, 1, 200)
    Eop_disp = calc_DP_TPR(sens_attr, y_true, score_optimal_Eop)
    auc_Eop = roc_auc_score(y_true, score_optimal_Eop)

    E_op_g1,E_op_g2 =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        E_op_g1, E_op_g2,_, _ = E_make(E_op_g1, E_op_g2,[], [], y_true,y_pred,  sens_attr)


    E_op_g1_calib,E_op_g2_calib =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in score_optimal_Eop])
        E_op_g1_calib, E_op_g2_calib,_, _ = E_make(E_op_g1_calib, E_op_g2_calib,[], [], y_true,y_pred,  sens_attr)
        

    AUC_init ,_, _ = AUC_make(y_true, y_score, sens_attr)
    AUC_Eop ,_, _ = AUC_make(y_true, score_optimal_Eop, sens_attr)


    ########################################  

    Eodd_disp = calc_EO_disp(sens_attr, y_true, score_optimal_Eodd)
    auc_Eodd = roc_auc_score(y_true, score_optimal_Eodd)
    AUC_Eod ,_, _ = AUC_make(y_true, score_optimal_Eodd, sens_attr)


    E_od_g1, E_od_g2 =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        _, _,E_od_g1, E_od_g2 = E_make([], [],E_od_g1, E_od_g2, y_true,y_pred,  sens_attr)
        
    E_od_g1_calib, E_od_g2_calib =[], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in score_optimal_Eodd])
        _, _,E_od_g1_calib, E_od_g2_calib = E_make([], [],E_od_g1_calib, E_od_g2_calib, y_true,y_pred,  sens_attr)
        


    ########################################  

    PR_disp = calc_DP_PR(sens_attr, y_true, score_optimal_PR)
    auc_PR = roc_auc_score(y_true, score_optimal_PR)


    AUC_PR ,_, _ = AUC_make(y_true, score_optimal_PR, sens_attr)


    PR_g1, PR_g2 = [], []

    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in y_score])
        _, PR_g1, PR_g2 = PR_make([], PR_g1, PR_g2,y_true, y_pred , sens_attr)



    PR_g1_calib, PR_g2_calib = [], []
    for TH in range:
        y_pred = np.array([1 if score > TH else 0 for score in score_optimal_PR])
        _, PR_g1_calib, PR_g2_calib = PR_make([], PR_g1_calib, PR_g2_calib,y_true, y_pred , sens_attr)










    L = 1.5
    F = 28
    F_legend = 22
    F_title = 32
    size = (8,6)




    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.fill_between(range, E_op_g1,E_op_g2, color='red', alpha=0.2, label = 'before: '+ str(round(100*Eop_disp_init,2)))
    plt.fill_between(range, E_op_g1_calib,E_op_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*Eop_disp,2)))
    plt.ylabel('EO', fontsize = F_title)
    plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
    plt.tight_layout()
    plt.legend(loc = 'lower left',fontsize = F_legend)
    plt.savefig('FIGURES/EO'+'_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()



    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.fill_between(range, E_od_g1,E_od_g2, color='red', alpha=0.2, label = 'before: '+ str(round(100*Eodd_disp_init,2)))
    plt.fill_between(range, E_od_g1_calib,E_od_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*Eodd_disp,2)))
    plt.ylabel('EOD', fontsize = F_title)
    plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
    plt.legend(fontsize = F_legend, loc = 'best')
    plt.tight_layout()
    plt.savefig('FIGURES/EOD_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()



    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.fill_between(range, PR_g2,PR_g1, color='red', alpha=0.2, label = 'before: '+ str(round(100*PR_disp_init,2)))
    plt.fill_between(range, PR_g1_calib,PR_g2_calib, color='blue', alpha=0.2, label = 'after: '+ str(round(100*PR_disp,2)))
    plt.ylabel('PR', fontsize = F_title)
    plt.xlabel('Threshold (' + r'$\tau$'+')', fontsize =F_title)  
    plt.legend(fontsize = F_legend, loc = 'best')
    plt.tight_layout()
    plt.savefig('FIGURES/PR_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()




    plt.figure(figsize=size)
    plt.xticks(fontsize = F)
    plt.yticks(fontsize = F)
    plt.plot(AUC_init[0],AUC_init[1],label = 'initial: '+str(round(100*auc_init,2)), color = 'black', linewidth = L) 
    plt.plot(AUC_PR[0],AUC_PR[1],label = 'PR: '+str(round(100*auc_PR,2)), color = 'red', linewidth = L) 
    plt.plot(AUC_Eop[0],AUC_Eop[1],label = 'EO: '+str(round(100*auc_Eodd,2)), color = 'green', linewidth = L) 
    plt.plot(AUC_Eod[0],AUC_Eod[1],label = 'EOD: '+str(round(100*auc_Eop,2)), color = 'blue', linewidth = L) 
    plt.legend(fontsize = F_legend, loc = 'best')
    plt.xlabel('FPR', fontsize =F_title)  
    plt.ylabel('TPR', fontsize =F_title)    
    plt.tight_layout()
    plt.savefig('FIGURES/AUC_'+str(model)+'_'+str(dataset)+'.pdf')
    plt.close()

