import os
import json

def get_infos(model_name, dataset):
    path = f'./infos/{model_name}/{dataset}'
    info_ls = []
    files = os.listdir(path)
    for file in files:
        if 'track' == file: continue
        with open(os.path.join(path, file), encoding='utf-8') as f:
            ls = json.load(f)
        info_ls.append(ls)
    return info_ls, files

datasets = ['FB15k237', 'wn18rr', 'YAGO3-10', 'family', 'umls', 'kinship']  
models = ['RotatE', 'ComplEx', 'HousE', 'TuckER', 'pLogicNet','RNNLogic']

for data in datasets:
    for model in models:
        try:
            infos_ls, files = get_infos(model, data)
        except:
            continue
        
        for infos, file in zip(infos_ls, files):
            n_info = []
            for info in infos:
                query, mode, rank, *_ = info
                n_info.append([query, mode, rank])
            os.makedirs(f'./n_infos/{model}/{data}', exist_ok=True)
            with open(f'./n_infos/{model}/{data}/{file}', 'w') as f:
                json.dump(n_info, f, indent=2)