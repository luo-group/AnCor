import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from scipy.stats import bootstrap
import numpy as np
import os
from Bio import SeqIO
import copy

def get_quantile(data, wt):
    data = pd.concat([data,pd.DataFrame({'mutant':"wt", 'DMS_score':wt}, index=[len(data)+1])], axis=0)
    data = data.sort_values(by='DMS_score', ascending=False).reset_index(drop=True)
    index = data[data.DMS_score == wt].index.tolist()[0]
    quantile = index/len(data)
    return index, quantile


def cut_seqlen(data, cut_len):
    for i in range(len(data)):
        data.loc[i, 'mutated_sequence'] = data.loc[i, 'mutated_sequence'][cut_len:]
        pre_pos = data.loc[i, 'mutated_position']
        if type(pre_pos) != str:
            data.loc[i, 'mutated_position'] = pre_pos-cut_len
        else:
            temp_pos = []
            for u in pre_pos.split(','):
                temp_pos.append(str(int(u)-cut_len))
            data.loc[i, 'mutated_position'] = ','.join(temp_pos)
    return data



class Mutation_Set_ProtenGym(Dataset):
    def __init__(self, data, fname, tokenizer, seq_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.wt_stat_file = pd.read_excel('data/wt_summary.xlsx')
        self.cut_off = float(self.wt_stat_file[self.wt_stat_file['protein_dataset'] == fname]['wt_fitness'])
        
        # wt_path = os.path.join('data', 'wt', f'{fname}.csv')
        # wt = pd.read_csv(wt_path)
        wt = self.wt_stat_file[self.wt_stat_file['protein_dataset'] == fname]['seq'].values[0]

        if len(wt) > seq_len:
            cut_len = len(wt) - seq_len
            print(f'===={fname} cut len :{cut_len}!=====')
            wt = wt[cut_len:]
            self.data = cut_seqlen(self.data, cut_len)


        self.seq, self.attention_mask = tokenizer(list(self.data['mutated_sequence']), padding=False,
                                                  truncation=True,
                                                  max_length=self.seq_len).values()

        target = [wt]*len(self.data)
        self.target, self.tgt_mask = tokenizer(target, padding=False, truncation=True,
                                               max_length=self.seq_len).values()
        self.score = torch.tensor(np.array(self.data['DMS_score']))
        self.pid = np.asarray(data['PID'])

        # if type(list(self.data['mutated_position'])[0]) != str:
        #     self.position = [[u] for u in self.data['mutated_position']]

        # else:
        temp = [str(u).split(',') for u in self.data['mutated_position']]
        self.position = []
        for u in temp:
            pos = [int(v) for v in u]
            self.position.append(pos)

    def __getitem__(self, idx):
        return [self.seq[idx], self.attention_mask[idx], self.target[idx],self.tgt_mask[idx] ,self.position[idx], self.score[idx], self.pid[idx]]

    def __len__(self):
        return len(self.score)

    def collate_fn(self, data):
        seq = torch.tensor(np.array([u[0] for u in data]))
        att_mask = torch.tensor(np.array([u[1] for u in data]))
        tgt = torch.tensor(np.array([u[2] for u in data]))
        tgt_mask = torch.tensor(np.array([u[3] for u in data]))
        pos = [torch.tensor(u[4]) for u in data]
        score = torch.tensor(np.array([u[5] for u in data]), dtype=torch.float32)
        pid = torch.tensor(np.array([u[6] for u in data]))
        return seq, att_mask, tgt, tgt_mask, pos, score, pid
    

    
def sample_data(dataset_name, seed, shot, frac=0.2):
    '''
    sample the train data and test data
    :param seed: sample seed
    :param frac: the fraction of testing data, default to 0.2
    :param shot: the size of training data
    '''

    data = pd.read_csv(f'data/{dataset_name}/data.csv', index_col=0)
    test_data = data.sample(frac=frac, random_state=seed)
    train_data = data.drop(test_data.index)
    kshot_data = train_data.sample(n=shot, random_state=seed)
    assert len(kshot_data) == shot, (
        f'expected {shot} train examples, received {len(train_data)}')

    kshot_data.to_csv(f'data/{dataset_name}/train.csv')
    test_data.to_csv(f'data/{dataset_name}/test.csv')


def split_train(dataset_name, shot, seed):
    '''
    five equal split training data, one of which will be used as validation set when training ConFit
    '''
    train = pd.read_csv(f'data/proteingym/{dataset_name}/train_{shot}shot_seed{seed}.csv', index_col=0)
    tlen = int(np.ceil(len(train) / 5))
    start = 0
    for i in range(1, 5):
        csv = train[start:start + tlen]
        start += tlen
        csv.to_csv(f'data/proteingym/{dataset_name}/train_{shot}shot_seed{seed}_{i}.csv')
    csv = train[start:]
    csv.to_csv(f'data/proteingym/{dataset_name}/train_{shot}shot_seed{seed}_{5}.csv')










