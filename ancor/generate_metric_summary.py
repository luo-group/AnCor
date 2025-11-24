import os

import torch
import pandas as pd
import numpy as np
from stat_utils import spearman, ndcg
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='aggregate result!')
parser.add_argument('--shot', type=int, default=72,
                    help='the shot num')
parser.add_argument('--method', type=str, help='algorithm name', default='ancor_esmc')
parser.add_argument('--dataset', type=str, help='dataset name', required=True)
parser.add_argument('--prefix', type=str, help='prefix name', default='ancor')
args = parser.parse_args()

if os.path.exists(f'results/summary_{args.method}_{args.shot}_{args.prefix}.xlsx'):
    summary = pd.read_excel(f'results/summary_{args.method}_{args.shot}_{args.prefix}.xlsx')
else:
    summary = pd.DataFrame(None)
    
wt_stat = pd.read_excel(f'data/wt_summary.xlsx')

for rpt in range(5):
    fname = args.dataset
    if os.path.exists(f'predicted_{args.method}/{fname}/{args.shot}_repeat{rpt}/pred_{args.prefix}.csv'):
        cut_off = float(wt_stat[wt_stat['protein_dataset'] == fname]['wt_fitness'])
        pred = pd.read_csv(f'predicted_{args.method}/{fname}/{args.shot}_repeat{rpt}/pred_{args.prefix}.csv')
        pred = pred.drop_duplicates(subset='PID')
        cn = []
        for i in range(1, 6):
            if f'{i}_x' in pred.columns:
                cn.append(f'{i}_x')
            elif f'{i}' in pred.columns:
                cn.append(f'{i}')

        if len(cn) < 5:
            print(f'Warining! not enough ensemble models: {fname}, repeat:{rpt}: {cn}')
        temp = pred[cn]
        temp = temp.mean(axis=1)
        pred = pd.concat([pred, temp], axis=1)
        pred = pred.rename(columns={0: 'avg'})
        test = pd.read_csv(f'data/proteingym/{fname}/test.csv', index_col=0)
        avg = pred[['avg', 'PID']]
        label = test[['PID', 'DMS_score', 'DMS_score_bin']]
        perf = pd.merge(avg, label, on='PID')
        score = list(perf['avg'])
        gscore = list(perf['DMS_score'])
        score = np.asarray(score)
        gscore = np.asarray(gscore)

        sndcg_q = ndcg(score, gscore, quantile=True, top=10)    #ndcg at quantile 0.1
        sndcg10 = ndcg(score, gscore, quantile=False, top=10)   #ndcg at rank 10
        

        out = pd.DataFrame({'dataset':fname,'ndcg10':sndcg10, 'ndcg_01':sndcg_q, 'method':f'{args.prefix}', 'repeat_seed':rpt}, index=[f'{fname}_{rpt}'])
        summary = pd.concat([summary, out], axis=0)

if not os.path.exists('results'):
    os.makedirs('results', exist_ok=True)
summary.to_excel(f'results/summary_{args.method}_{args.shot}_{args.prefix}.xlsx', index=False)

