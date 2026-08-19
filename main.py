import argparse
import os
import json
from utils import *
#from vistool import VisTool
from DEMO_PROBE import DEMO_PROBE

def find_data(data_path: str):
    data_here = os.path.exists(data_path)
    if not data_here:
        # request data folder
        data_path = ...
    else:
        pass
    
    e2id, id2e = get_e2id(data_path)
    r2id, id2r = get_r2id(data_path)
    nentity = len(e2id)
    nrelation = len(r2id)
    trn_triples, val_triples, tst_triples = get_triples(data_path, e2id, r2id)
    count_info_dict_trn = count_entities(trn_triples)
    
    return e2id, id2e, r2id, id2r, nentity, nrelation, trn_triples, tst_triples, count_info_dict_trn

def find_baseline_models(data: str):
    selected_baseline_models = list()
    
    baseline_models = os.listdir('./n_infos')
    for baseline in baseline_models:
        datas = os.listdir(f'./n_infos/{baseline}')
        if data in datas: selected_baseline_models.append(baseline)
    
    return selected_baseline_models, len(selected_baseline_models) > 0

def get_baseline_infos(data: str, baseline_models: list):
    baseline_infos = [None] * len(baseline_models)
    for b_i, base in enumerate(baseline_models):
        files = os.listdir(f'./n_infos/{base}/{data}')
        baseline_infos[b_i] = [None] * len(files)
        for f_i, file in enumerate(files):
            with open(f'./n_infos/{base}/{data}/{file}', 'r') as f:
                baseline_infos[b_i][f_i] = json.load(f)
    return baseline_infos

def main():
    user_data = 'FB15k237' # user inputed data name
    user_model = 'RotatE' # user inputed model name
    
    # search local if the data is already there. if so, fetch, else, request user to dump the data folder
    e2id, id2e, r2id, id2r, nentity, nrelation, trn_triples, tst_triples, count_info_dict_trn = find_data(f'./data/{user_data}')
    
    # search inside n_infos/{model}/{data}  for baseline models that already have results of that data 
    baseline_models, base_flag = find_baseline_models(user_data)
    nbase = len(baseline_models)
    
    # fetch infos from user upload
    user_model_infos = None
    
    # fetch infos from baseline models
    '''
    Informations inside n_infos folder as json format.
    It is a 2d list where each element is [[h, r, t], mode, rank] where mode is either 'h' or 't'
    '''
    baseline_infos = None
    if base_flag:
        baseline_infos = get_baseline_infos(user_data, baseline_models)
    
    # define PROBE for both user_model and baeline_models. The first index of met_ls is the user model.
    met_ls = [None] * (nbase + 1)
    
    user_model_infos = baseline_infos[0] # just for example
    met_ls[0] = [DEMO_PROBE(model=user_model, data=user_data, nentity=nentity, info_dump=info,
                            count_info_dict_trn=count_info_dict_trn, alpha=1, beta=0,) for
                            info in user_model_infos]
    
    for b_i, (model, base_info_ls) in enumerate(zip(baseline_models, baseline_infos)):
        met_ls[b_i+1] = [DEMO_PROBE(model=model, data=user_data, nentity=nentity, info_dump=info,
                            count_info_dict_trn=count_info_dict_trn, alpha=1, beta=0,) for
                            info in base_info_ls]
    
    alphas = [1.0, 0.5, 0.25] # beta = 0.0
    betas = [0.0, 0.4, 0.8]   # alpha = 1.0
    
    '''
    1. individual bar plot
    given a pair of (alpha, beta), it should create a figure where the bar height is
    the model performance. each figure should be individual plots for different (alpha, beta) pair.
    '''
    



if __name__ == '__main__':
    main()