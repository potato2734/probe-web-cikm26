import numpy as np
import math
from collections import defaultdict


class DEMO_PROBE():
    def __init__(self, model, data, nentity, info_dump, count_info_dict_trn, alpha, beta):
        self.model = model
        self.data = data
        self.nentity = nentity
        self.info = info_dump
        self.count_info_dict_trn = count_info_dict_trn
        self.query, self.mode, self.rank, self.p_e = self.unpack_info(info_dump)
        self.alpha = alpha
        self.beta = beta

    def unpack_info(self, info):
        query_ls = []
        mode_ls = []
        rank_ls = []
        p_e_ls = []
        
        for query_info in info:
            query, mode, rank = query_info
            query_ls.append(query)
            mode_ls.append(mode)
            target_e = query[0] if mode == 'h' else query[2]
            
            rank_ls.append(rank)
            p_e_ls.append(self.count_info_dict_trn[target_e])
            
        return query_ls, mode_ls, np.array(rank_ls), np.array(p_e_ls)

    def set_transform_function(self, alpha, mode='f*'):
        self.alpha = alpha
        if mode == 'f':
            self.transformed_ranks = (1 / self.rank) ** self.alpha
            return
        elif mode == 'f*':
            N_coeff = np.array([self.nentity] * len(self.query))
            N_coeff = (1 / N_coeff) ** self.alpha
            self.transformed_ranks = ((1 / self.rank) ** self.alpha - N_coeff) / (1 - N_coeff)
        else:
            raise Exception(f'No mode supports \'{mode}\'')

    @staticmethod
    def normalize_array(arr):
        norm_arr = arr / np.sum(arr)
        assert math.isclose(sum(norm_arr), 1.0, rel_tol=1e-10)
        return norm_arr

    def w_entity_function(self, arr, param, epsilon):
        return 1 / (epsilon + arr) ** param

    def set_entity_weight(self, beta):
        self.beta = beta
        self.entity_raw_weights = self.w_entity_function(self.p_e, self.beta, 1)


    def calculate_final_metric(self, alpha, beta, tmode='f*'):
        self.alpha = alpha
        self.beta = beta

        self.set_transform_function(alpha, mode=tmode)
        self.set_entity_weight(self.beta)

        self.norm_W = DEMO_PROBE.normalize_array(self.entity_raw_weights)

        assert len(self.transformed_ranks) == len(self.norm_W), print(len(self.transformed_ranks), len(self.norm_W))

        final_metric = np.dot(self.transformed_ranks, self.norm_W)

        return final_metric


