import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from sklearn.metrics import ndcg_score

###### statistical metrics functions #####

def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]

def minmax(x):
    return ( (x - np.min(x)) / (np.max(x) - np.min(x)) ) 

def ndcg(y_true, y_score, **kwargs):
    '''
    borrowed and modified from the ProteinGym official repo:https://github.com/OATML-Markslab/ProteinGym
    Inputs:
        y_true: an array of the true scores where higher score is better
        y_score: an array of the predicted scores where higher score is better
    Options:
        quantile: If True, uses the top k quantile of the distribution
        top: under the quantile setting this is the top quantile to
            keep in the gains calc. This is a PERCENTAGE (i.e input 10 for top 10%)
            if choose all, then calculate the whole seq ndcg

    '''
    if 'quantile' not in kwargs:
        kwargs['quantile'] = False
    if 'top' not in kwargs:
        kwargs['top'] = 10
    if kwargs['quantile']:
        k = np.floor(y_true.shape[0]*(kwargs['top']/100)).astype(int)
    else:
        k = kwargs['top']
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    gains = minmax(y_true)
    ranks = np.argsort(np.argsort(-y_score)) + 1
    
    if kwargs['top'] == 'all':
        k = len(ranks)
    #sub to top k
    ranks_k = ranks[ranks <= k]
    gains_k = gains[ranks <= k]
    #all terms with a gain of 0 go to 0
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]
    
    #if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return (0)
    
    #discounted cumulative gains
    dcg = np.sum([g/np.log2(r+1) for r,g in zip(ranks_fil, gains_fil)])
    
    #ideal dcg - calculated based on the top k actual gains
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= k]
    ideal_gains_k = gains[ideal_ranks <= k]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum([g/np.log2(r+1) for r,g in zip(ideal_ranks_fil, ideal_gains_fil)])
    
    #normalize
    ndcg = dcg/idcg
    
    return (ndcg)

def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    return mean, std


##### score functions #####

def mask_marginal(model, seq, mask, wt, wt_mask, pos, tokenizer):
    '''
    compute mutational proxy using masked marginal probability
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device

    mask_seq = seq.clone()
    m_id, _ = tokenizer('<mask>', add_special_tokens=False).values()
    m_id = m_id[0]

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos+1] = m_id


    logits = model(mask_seq, mask).sequence_logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)

    for i in range(batch_size):

        scores[i] = torch.sum(log_probs[i][pos[i]+1, seq[i][pos[i]+1]])-torch.sum(log_probs[i][pos[i]+1, wt[i][pos[i]+1]])


    return scores, logits


def mask_marginal_1v(model, seq, mask, wt, wt_mask, pos, tokenizer):
    '''
    compute mutational proxy using masked marginal probability, designed for ESM-1v and ESM2 models
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device

    mask_seq = seq.clone()
    m_id, _ = tokenizer('<mask>', add_special_tokens=False).values()
    m_id = m_id[0]

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos+1] = m_id



    logits = model(mask_seq, mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)

    for i in range(batch_size):

        scores[i] = torch.sum(log_probs[i][pos[i]+1, seq[i][pos[i]+1]])-torch.sum(log_probs[i][pos[i]+1, wt[i][pos[i]+1]])


    return scores, logits

def wt_marginal(model, seq, mask, wt, wt_mask, pos, tokenizer):
    '''
    compute mutational proxy using wt marginal probability
    :param seq:mutant seq
    :param wt: wild type sequence
    :param wt_mask: mask for input wt
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device

    mask_seq = seq.clone()

    batch_size = int(seq.shape[0])

    # ### for the same protein, the wt is always the same, so we only do 1 forward
    # wt = torch.unsqueeze(wt[0], 0)
    # wt_mask = torch.unsqueeze(wt_mask[0], 0)


    logits = model(wt, wt_mask).sequence_logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)
    # print(f'step 2, cached memory:{torch.cuda.memory_cached() / 1024 ** 2}MB')
    # print(f'step 2, reserved memory:{torch.cuda.memory_reserved() / 1024 ** 2}MB')
    # print(f'step 2, used memory:{torch.cuda.memory_allocated() / 1024 ** 2}MB')

    for i in range(batch_size):

        # mut_pos = pos[i]
        # score_i = log_probs[i]
        # wt_i = wt[i]
        # seq_i = seq[i]
        scores[i] = torch.sum(log_probs[i][pos[i]+1, seq[i][pos[i]+1]])-torch.sum(log_probs[i][pos[i]+1, wt[i][pos[i]+1]])

    # print(f'step 3, used memory:{torch.cuda.memory_allocated() / 1024 ** 2}MB')

    return scores, logits

def Pseudo_likelihood(model, seq, mask, wt, wt_mask, pos, tokenizer):
    '''
    compute mutational proxy using Pseudolikelihood
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param wt_mask: mask for input wt
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device

    mask_seq = seq.clone()

    batch_size = int(seq.shape[0])

    ### for the same protein, the wt is always the same, so we only do 1 forward
    wt = torch.unsqueeze(wt[0], 0)
    wt_mask = torch.unsqueeze(wt_mask[0], 0)

    out = model(mask_seq, mask)
    out_wt = model(wt, wt_mask)
    logits = out.sequence_logits
    log_probs = torch.log_softmax(logits, dim=-1)
    logits_wt = out_wt.sequence_logits
    log_probs_wt = torch.log_softmax(logits_wt, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)

    for i in range(batch_size):

        mut_pos = pos[i]
        score_i = log_probs[i]
        score_wt_i = log_probs_wt[0]
        wt_i = wt[0]
        seq_i = seq[i]
        scores[i] = torch.sum(score_i[mut_pos+1, seq_i[mut_pos+1]])-torch.sum(score_wt_i[mut_pos+1, wt_i[mut_pos+1]])

    return scores, logits



### loss functions ### 
    
def ancor_loss(policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    ancor_direction,
                    beta,
                    shot, 
                    reference_free=False,
                    is_eval=False
):
    """Compute the AnCor loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        ancor_direction: direction tensor specify the ancor loss between (ancor_up, ancor_cross, ancor_down). Shap:(batch_size,3)
        beta: Temperature parameter for the sigmoid, typically something in the range of 0.1 to 0.5. we chose 0.1 in the AnC
        shot: number of training size, used to donwsampling the anchor seq.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses. We chose False in the AnCor.
    Returns:
        If eval, A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        else: the losses tensor.
    """


    if reference_free:
        chosen_logratios = policy_chosen_logps
        rejected_logratios = policy_rejected_logps

    else:

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

    weight = 3/shot


    losses_chosen = torch.log(F.sigmoid(beta * chosen_logratios))  # chosen likelihood
    losses_rejected = torch.log(F.sigmoid(beta * rejected_logratios))  # rejected likelihood

    losses_rel = torch.log(F.sigmoid(
        beta * (chosen_logratios - rejected_logratios)
    ))      # increase gap between chosen and rejected, calibrate the relative preference

    cross_losses =  -weight*losses_chosen + weight*losses_rejected - (3-weight)*losses_rel   # chosen>anchor>rejected

    down_losses = weight*losses_chosen + weight*losses_rejected - (3-weight)*losses_rel    # anchor>chosen>rejected

    up_losses = -weight*losses_rejected - weight*losses_chosen - (3-weight)*losses_rel      # chosen>rejected>anchor

    all_loss = torch.stack([up_losses, cross_losses, down_losses], dim=0)


    losses = torch.stack([up_losses, cross_losses, down_losses], dim=0).T * ancor_direction
    loss = losses.sum(dim=1)

    if is_eval:
        return losses_chosen.detach().clone().mean(), losses_rejected.detach().clone().mean(), losses_rel.detach().clone().mean(), loss.mean()

    else:
        return loss.mean()
    



def sample_ancor_pairs(policy_scores, reference_scores, golden_score, cut_off):
    batch_size = policy_scores.shape[0]
    device = policy_scores.device
    ls_size = int(batch_size*(batch_size-1)/2)
    policy_chosen_score = torch.zeros((ls_size,), dtype=torch.float32, device=device)
    policy_reject_score = torch.zeros((ls_size,), dtype=torch.float32, device=device)
    reference_chosen_score = torch.zeros((ls_size,), dtype=torch.float32, device=device)
    reference_reject_score = torch.zeros((ls_size,), dtype=torch.float32, device=device)
    ancor_direction = []
    idx = -1
    for i in range(int(batch_size)):
        for j in range(i+1, int(batch_size)):
            idx += 1
            if golden_score[i] > golden_score[j]:
                policy_chosen_score[idx] = policy_scores[i]
                policy_reject_score[idx] = policy_scores[j]
                reference_chosen_score[idx] = reference_scores[i]
                reference_reject_score[idx] = reference_scores[j]
                ancor_direction.append([golden_score[i]>cut_off and golden_score[j]>cut_off, golden_score[i]>cut_off and golden_score[j]<=cut_off, golden_score[i]<=cut_off and golden_score[j]<=cut_off])
            else:
                policy_chosen_score[idx] = policy_scores[j]
                policy_reject_score[idx] = policy_scores[i]
                reference_chosen_score[idx] = reference_scores[j]
                reference_reject_score[idx] = reference_scores[i]
                ancor_direction.append([golden_score[j]>cut_off and golden_score[i]>cut_off, golden_score[j]>cut_off and golden_score[i]<=cut_off, golden_score[j]<=cut_off and golden_score[i]<=cut_off])
            
    return policy_chosen_score, policy_reject_score, reference_chosen_score, reference_reject_score, torch.tensor(ancor_direction, dtype=torch.float32, device=device)



def BT_loss(scores, golden_score):
    loss = torch.tensor(0.)
    loss = loss.cuda()
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    return loss




def KLloss(logits, logits_reg, seq, att_mask):

    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]


        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        loss += creterion_reg(reg.log(), pred)
    return loss



