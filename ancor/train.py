import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from functools import cache, partial
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig, AutoModelForMaskedLM, AutoTokenizer
from peft.utils.other import fsdp_auto_wrap_policy
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMCInferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
)
from esm.tokenization import (
    get_esm3_model_tokenizers,
    get_esmc_model_tokenizers,
)
import os
import argparse
from pathlib import Path
import accelerate
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from safetensors import safe_open

from data_utils import Mutation_Set_ProtenGym, split_train
from stat_utils import *
import gc
import warnings
import time
import yaml
warnings.filterwarnings("ignore")


Score_MAP = {
    "mask_marginal": mask_marginal,
    "wt_marginal": wt_marginal,
    "pseudo_likelihood": Pseudo_likelihood
}

@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    # Try to download from hugginface if it doesn't exist
    if model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path



def train(model, model_reg, trainloder, optimizer, tokenizer, cut_off, beta, kl_coefficient, score_func, training_size):

    model.train()

    total_loss = 0.

    for step, data in enumerate(trainloder):
        seq, mask = data[0], data[1]
        wt, wt_mask = data[2], data[3]
        pos = data[4]
        golden_score = data[5]
        score, policy_logits = score_func(model, seq, mask, wt, wt_mask, pos, tokenizer)
        with torch.no_grad():
            ref_score, ref_logits = score_func(model_reg, seq, mask, wt, wt_mask, pos, tokenizer)
        # score = score.cuda()
        policy_chosen_score, policy_reject_score, reference_chosen_score, reference_reject_score, ancor_direction = sample_ancor_pairs(score, ref_score, golden_score, cut_off)

        loss = ancor_loss(policy_chosen_score, policy_reject_score, reference_chosen_score, reference_reject_score, ancor_direction, beta, training_size, reference_free=True)
        loss_kl = KLloss(policy_logits, ref_logits, seq, mask)
        loss += kl_coefficient * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate(model, testloader, tokenizer, accelerator, score_func, istest=False):
    model.eval()
    seq_list = []
    score_list = []
    gscore_list = []
    with torch.no_grad():
        for step, data in enumerate(testloader):
            seq, mask = data[0], data[1]
            wt, wt_mask = data[2], data[3]
            pos = data[4]
            golden_score = data[5]
            pid = data[6]
            if istest:
                pid = pid.cuda()
                pid = accelerator.gather(pid)
                for s in pid:
                    seq_list.append(s.cpu())

            score, logits = score_func(model, seq, mask, wt, wt_mask, pos, tokenizer)
            score = score.cuda()
            score = accelerator.gather(score)
            golden_score = accelerator.gather(golden_score)
            score = np.asarray(score.cpu())
            golden_score = np.asarray(golden_score.cpu())
            score_list.extend(score)
            gscore_list.extend(golden_score)
    score_list = np.asarray(score_list)
    gscore_list = np.asarray(gscore_list)
    sr = ndcg(score_list, gscore_list)

    if istest:
        seq_list = np.asarray(seq_list)

        return sr, score_list, seq_list
    else:
        return sr, logits[0]


def main():
    parser = argparse.ArgumentParser(description='ConFit train, set hyperparameters')
    parser.add_argument('--config', type=str, default='48shot_config.yaml',
                        help='the config file name')
    parser.add_argument('--dataset', type=str, help='the dataset name')
    parser.add_argument('--sample_seed', type=int, default=0, help='the sample seed for dataset')
    parser.add_argument('--model_seed', type=int, default=1, help='the random seed for the pretrained model initiate')
    parser.add_argument('--shot', type=int, default=48, help='the training data size')
    parser.add_argument('--measure', type=str, help='the metrics name, please choose from ee or Yield', default='ee')
    parser.add_argument('--score', type=str, help='the socre function, please choose from mask_marginal, wt_marginal, and pseudo_likelihood', default='wt_marginal')
    parser.add_argument('--prefix', type=str, help='the training prefix', default='esmc_wt')
    parser.add_argument('--kl', type=float, default=0.2, help='the kl coefficient')
    parser.add_argument('--log_logits', action='store_true', help='whether to log logits')
    args = parser.parse_args()
    dataset = args.dataset

    #read in config
    with open(f'{args.config}', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    batch_size = int(int(config['batch_size'])/int(config['gpu_number']))
    beta = config['beta']
    kl_coefficient = config['kl_coefficient']


    accelerator = Accelerator()

    if config['model'] == 'ESMC':

        ### creat model
        model = ESMC(d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=False,
            )
        

        tokenizer = get_esmc_model_tokenizers()

        state_dict = torch.load(
            data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
            map_location='cpu',
        )
        model.load_state_dict(state_dict=state_dict)

        model_reg = ESMC(d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=True,
            ).eval()
        model_reg.load_state_dict(state_dict=state_dict)

        

    else:
        AttributeError('curretly only support ESMC model!')


    for pm in model_reg.parameters():
        pm.requires_grad = False
    model_reg.eval()    #regularization model

    if config['model'] == 'ESMC':
        target_modules = ["layernorm_qkv.1", "out_proj", "query", "value", "dense"]
        peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        bias="none",
        target_modules=target_modules,
    )
    else:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(config['lora_r']),
            lora_alpha=int(config['lora_alpha']),
            lora_dropout=float(config['lora_dropout']),
            target_modules=["query", "value"]
        )

    model = get_peft_model(model, peft_config)



    # create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['ini_lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2*int(config['max_epochs']), eta_min=float(config['min_lr']))
    if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model_reg = accelerator.prepare(model_reg)

    accelerator.print(f'===================dataset:{dataset}, preparing data=============')



    with accelerator.main_process_first():
        train_csv = pd.DataFrame(None)
        split_train(dataset_name=args.dataset, shot=args.shot, seed=args.sample_seed)
        test_csv = pd.read_csv(f'data/proteingym/{args.dataset}/test.csv')
        
        for i in range(1, 6):
            if i == args.model_seed:
                val_csv = pd.read_csv(f'data/proteingym/{args.dataset}/train_{args.shot}shot_seed{args.sample_seed}_{i}.csv')   #using 1/5 train data as validation set
            else:
                temp_csv = pd.read_csv(f'data/proteingym/{args.dataset}/train_{args.shot}shot_seed{args.sample_seed}_{i}.csv')
                train_csv = pd.concat([train_csv, temp_csv], axis=0)
                train_csv = train_csv.reset_index(drop=True)


    #creat dataset and dataloader
    trainset = Mutation_Set_ProtenGym(data=train_csv, fname=dataset, tokenizer=tokenizer)
    testset = Mutation_Set_ProtenGym(data=test_csv, fname=dataset,  tokenizer=tokenizer)
    valset = Mutation_Set_ProtenGym(data=val_csv, fname=dataset,  tokenizer=tokenizer)
    cut_off = trainset.cut_off
    with accelerator.main_process_first():
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn, shuffle=True)
        testloader = DataLoader(testset, batch_size=2, collate_fn=testset.collate_fn)
        valloader = DataLoader(valset, batch_size=2, collate_fn=testset.collate_fn)

    trainloader = accelerator.prepare(trainloader)
    testloader = accelerator.prepare(testloader)
    valloader = accelerator.prepare(valloader)
    accelerator.print('==============data preparing done!================')
    # accelerator.print("Current allocated memory:", torch.cuda.memory_allocated())
    # accelerator.print("cached:", torch.cuda.memory_reserved())


    best_sr = -np.inf
    endure = 0
    best_epoch = 0
    # wt_score = trainset.wt_fitness
    score_func = Score_MAP[args.score]

    if args.log_logits:
        logits_dic = {}

    for epoch in range(int(config['max_epochs'])):
        loss = train(model, model_reg, trainloader, optimizer, tokenizer, cut_off, beta=beta, kl_coefficient=kl_coefficient, training_size=args.shot, score_func=score_func)
        accelerator.print(f'========epoch{epoch}; training loss :{loss}=================')
        sr, logits = evaluate(model, valloader, tokenizer, accelerator, score_func)
        accelerator.print(f'========epoch{epoch}; val ndcg :{sr}=================')
        if args.log_logits:
            logits_dic[epoch] = logits.cpu()
        scheduler.step()
        if best_sr >= sr:
            endure += 1
        else:
            endure = 0
            best_sr = sr
            best_epoch = epoch

            if not os.path.isdir(f'checkpoint_esmc/ancor_esmc/{dataset}_{args.shot}'):
                if accelerator.is_main_process:
                    os.makedirs(f'checkpoint_esmc/ancor_esmc/{dataset}_{args.shot}')
            save_path = os.path.join('checkpoint_esmc', f'ancor_esmc/{dataset}_{args.shot}',
                                     f'sample_seed{args.sample_seed}_model_seed{args.model_seed}_{args.prefix}')
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path)
        
        if endure > int(config['endure_time']):
            accelerator.print(f'========early stop at epoch{epoch}!============')
            break

    if args.log_logits:
        if accelerator.is_main_process:
            ### pop the logits that overpass the best epoch
            for k in range(best_epoch+1, epoch+1):
                logits_dic.pop(k)


    # inference on the test sest
    accelerator.print('=======training done!, test the performance!========')
    save_path = Path(os.path.join('checkpoint_esmc', f'ancor_esmc/{dataset}_{args.shot}', f'sample_seed{args.sample_seed}_model_seed{args.model_seed}_{args.prefix}'))
    del model
    accelerator.free_memory()

    if config['model'] == 'ESMC':

        model = ESMC(d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=False,
            )
        state_dict = torch.load(
            data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
            map_location='cpu',
        )
        model.load_state_dict(state_dict=state_dict)
        
    else:
        AttributeError('curretly only support ESMC model!')


    model = PeftModel.from_pretrained(model, save_path)
    model = accelerator.prepare(model)
    accelerator.wait_for_everyone()
    sr, score, pid = evaluate(model, testloader, tokenizer, accelerator, score_func, istest=True)
    pred_csv = pd.DataFrame({f'{args.model_seed}': score, 'PID': pid})
    if args.log_logits:
        if accelerator.is_main_process:
            if not os.path.isdir(f'logits_ancor_esmc/proteingym/{dataset}'):
                os.makedirs(f'logits_ancor_esmc/proteingym/{dataset}', exist_ok=True)
            torch.save(logits_dic, f'logits_ancor_esmc/proteingym/{dataset}/logits_{args.shot}shot_sample_seed{args.sample_seed}_model_seed{args.model_seed}_{args.prefix}.pt')
    if accelerator.is_main_process:
        if not os.path.isdir(f'predicted_ancor_esmc/{dataset}/{args.shot}_repeat{args.sample_seed}'):
            os.makedirs(f'predicted_ancor_esmc/{dataset}/{args.shot}_repeat{args.sample_seed}')
        if os.path.exists(f'predicted_ancor_esmc/{dataset}/{args.shot}_repeat{args.sample_seed}/pred_{args.prefix}.csv'):
            pred = pd.read_csv(f'predicted_ancor_esmc/{dataset}/{args.shot}_repeat{args.sample_seed}/pred_{args.prefix}.csv', index_col=0)
            pred = pd.merge(pred, pred_csv, on='PID')
        else:
            pred = test_csv
            pred = pd.merge(pred, pred_csv, on='PID')
        pred.to_csv(f'predicted_ancor_esmc/{dataset}/{args.shot}_repeat{args.sample_seed}/pred_{args.prefix}.csv')
    accelerator.print(f'=============the test ndcg for early stop: {sr}==================')



if __name__ == "__main__":
    main()





