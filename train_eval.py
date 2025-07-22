import time
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
from tqdm import *

# training function at each epoch

def train(loop, DDI_predictor, DDI_sampler, predictor_optim, sampler_optim, adj_matrix, edge_rel, avg_reward, args, dataset_statistics):
    total_loss1, total_loss2, total_reward = 0, 0, 0

    prob_all = []
    label_all = []

    last_reward = 0

    num_nodes = dataset_statistics['num_nodes']
    cnt = 0

    """Train one epoch"""
    for j, data in enumerate(loop):
        data_mol1 = data[0].cuda()
        data_mol2 = data[1].cuda()
        data_subgraph = data[2].cuda()
        data_idx = data[3].cuda() # batch_size * 2

        cnt += data_idx.shape[0]

        DDI_predictor.train()
        DDI_sampler.eval()
        predictor_optim.zero_grad()

        if args.extractor == 'RL':
            with torch.no_grad():
                embeddings = DDI_predictor.drug_node_feature.node_encoder(torch.tensor(range(num_nodes)).cuda())
                selected_subgraph_list, batch = DDI_sampler.predict(data_idx, adj_matrix, edge_rel, embeddings, data_subgraph) 
            selected_subgraph_list = selected_subgraph_list.cuda()
            batch = batch.cuda()
            predicts, loss, _ = DDI_predictor(data_mol1, data_mol2, selected_subgraph_list)
        else: # used default subgraph
            predicts, loss = DDI_predictor(data_mol1, data_mol2, data_subgraph)
        loss.backward()

        prob_all.append(predicts)
        label_all.append(data_mol1.y)

        total_loss1 += loss.item() * num_graphs(data_mol1)

        predictor_optim.step()

        if args.extractor == 'RL': 

            sampler_optim.zero_grad()
            DDI_predictor.eval()
            DDI_sampler.train()

            with torch.no_grad():
                batch = torch.tensor(range(data_idx.shape[0])).cuda()
                pred_default = DDI_predictor.pred(data_mol1, data_mol2, data_subgraph, batch)

            with torch.no_grad():
                embeddings = DDI_predictor.drug_node_feature.node_encoder(torch.tensor(range(num_nodes)).cuda())
            selected_subgraph_list, selected_subgraph_prob_list, batch, tmp = DDI_sampler(data_idx, adj_matrix, edge_rel, embeddings, data_subgraph)

            selected_subgraph_list = selected_subgraph_list.cuda()
            batch = batch.cuda()

            with torch.no_grad():
                reward_batch, predicts, = DDI_predictor.get_reward(data_mol1, data_mol2, selected_subgraph_list, batch, pred_default)
            
            reward_all = []
            for b in torch.unique(batch):
                idx = (batch == b).nonzero(as_tuple=True)[0]
                batch_reward = reward_batch[idx]
                reward = torch.zeros(batch_reward.size(), device=batch_reward.device)
                R = 0
                n = batch_reward.size(0) - 1
                for i, r in enumerate(batch_reward.flip(0)):
                    R = r + 0.9 * R
                    reward[n-i] = R
                reward_all.append(reward)
                last_reward += reward[-1].item()
            reward_batch = torch.concat(reward_all)

            total_reward += torch.sum(reward_batch).item()

            reinforce_loss = -1 * torch.sum(reward_batch * selected_subgraph_prob_list)
            total_loss2 += reinforce_loss.item()
            reinforce_loss.backward()
            sampler_optim.step()

    avg_reward = total_reward / len(loop)
    last_reward = last_reward / cnt
    avg_loss = total_loss1 / len(loop)

    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    train_acc, train_f1, train_auc, train_aupr = get_score(label_all, prob_all)
    return train_acc, train_f1, train_auc, train_aupr, avg_loss, avg_reward


def eval(loader, DDI_predictor, DDI_sampler, adj_matrix, edge_rel, dataset_statistics, args):
    total_reward, total_loss = 0, 0
    DDI_predictor.eval()
    DDI_sampler.eval()

    prob_all = []
    label_all = []

    num_nodes = dataset_statistics['num_nodes']

    with torch.no_grad():
        for idx, data in enumerate(loader):
            data_mol1 = data[0].cuda()
            data_mol2 = data[1].cuda()
            data_subgraph = data[2].cuda()
            data_idx = data[3].cuda() 


            if args.extractor != 'RL':
                predicts, loss, att_weights = DDI_predictor(data_mol1, data_mol2, data_subgraph)
            else:
                batch = torch.tensor(range(data_idx.shape[0])).cuda()
                pred_default = DDI_predictor.pred(data_mol1, data_mol2, data_subgraph, batch)

                embeddings = DDI_predictor.drug_node_feature.node_encoder(torch.tensor(range(num_nodes)).cuda())
                selected_subgraph_list, batch = DDI_sampler.predict(data_idx, adj_matrix, edge_rel, embeddings, data_subgraph) 
                selected_subgraph_list = selected_subgraph_list.cuda()
                batch = batch.cuda()
                
                reward_batch, predicts = DDI_predictor.get_reward(data_mol1, data_mol2, selected_subgraph_list, batch, pred_default)
                total_reward += torch.sum(reward_batch).item()


            prob_all.append(predicts)
            label_all.append(data_mol1.y)

    eval_loss = total_loss / len(loader.dataset)
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    eval_acc, eval_f1, eval_auc, eval_aupr = get_score(label_all, prob_all)

    avg_reward = total_reward / len(loader.dataset)

    return eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss, avg_reward


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.c_size

def get_score(label_all, prob_all):

    predicts_label = [1 if prob >= 0.5 else 0 for prob in prob_all]

    acc = accuracy_score(label_all, predicts_label)
    f1 = f1_score(label_all, predicts_label)
    auroc = roc_auc_score(label_all, prob_all)
    p, r, t = precision_recall_curve(label_all, prob_all)
    auprc = auc(r, p)

    return acc, f1, auroc, auprc


