import os
import argparse
import gc
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import json
import copy
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from model.predictor import Predictor, Predictor2
from model.sampler import Sampler
from torch_geometric.utils import degree
from torch.utils.data.distributed import DistributedSampler
from data_process import smile_to_graph, read_smiles, read_interactions, generate_node_subgraphs, read_network, process_node_graph, read_ppr
from sklearn.model_selection import StratifiedKFold, KFold
from train_eval import train, test, eval
import random
import pickle

import pdb
import sys

# def custom_except_hook(exc_type, exc_value, exc_traceback):
#     print(f"\nException caught: {exc_value}")
#     pdb.post_mortem(exc_traceback)

# sys.excepthook = custom_except_hook

def init_args(user_args=None):

    parser = argparse.ArgumentParser(description='TIGER')

    parser.add_argument('--model_name', type=str, default='tiger')

    parser.add_argument('--dataset', type=str, default="drugbank")

    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--layer', type=int, default=2)

    parser.add_argument('--predictor_lr', type=float, default=0.001)
    parser.add_argument('--sampler_lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eps', type=float, default=5e-6)
    parser.add_argument('--no_tqdm', action='store_true')


    parser.add_argument('--pos', type=float, default=3)
    parser.add_argument('--neg', type=float, default=1)


    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--extractor', type=str, default="randomWalk") ##option [khop-subtree, randomWalk, probability]
    parser.add_argument('--graph_fixed_num', type=int, default=1)
    parser.add_argument('--khop', type=int, default=1)
    parser.add_argument('--fixed_num', type=int, default=32)

    parser.add_argument('--mode', type=str, default="s1") 
    parser.add_argument('--load_predictor', type=str, default="./best_save/tiger/drugbank/khop-subtree/s3/fold_0/0.85018/DDI_predictor.pt")
    parser.add_argument('--load_sampler', type=str, default="")

    # Graphormer
    parser.add_argument("--d_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_smiles_degree", type=int, default=300)
    parser.add_argument("--max_graph_degree", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--k_step", type=int, default=10)

    # Linkformer
    parser.add_argument("--num_heads_l", type=int, default=1)
    parser.add_argument("--num_layers_l", type=int, default=1)


    # coeff
    parser.add_argument('--sub_coeff', type=float, default=0.1)
    parser.add_argument('--mi_coeff', type=float, default=0.1)

    parser.add_argument('--s_type', type=str, default='random')

    args = parser.parse_args()

    return args


def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def k_fold(data, kf, folds, y):

    test_indices = []
    train_indices = []

    if len(y):
        for _, idx in kf.split(torch.zeros(len(data)), y):
            test_indices.append(idx)
    else:
        for _, idx in kf.split(data):
            test_indices.append(idx)

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def split_fold(folds, dataset, labels, args, scenario_type='random'):

    test_indices, train_indices, val_indices = [], [], []

    if scenario_type == 'random':
        skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
        train_indices, test_indices, val_indices = k_fold(dataset, skf, folds, labels)
    elif scenario_type == 'inductive':
        with open(f'data/{args.dataset}/inductive_split.pkl', 'rb') as f:
            data = pickle.load(f)
        train_indices = data['train']
        test_indices = data['test']
        val_indices = data['val']

    return train_indices, test_indices, val_indices

def load_data(args):


    dataset = args.dataset

    data_path = "/bigdat2/user/xiejc/zhangc/dataset/TIGER/dataset/" + dataset + "/"

    ligands = read_smiles(os.path.join(data_path, "drug_smiles.txt"))


    # smiles to graphs
    print("load drug smiles graphs!!")
    smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

    print("load networks !!")
    num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "networks.txt")
    num_node += 1

    print("load DDI samples!!")
    interactions_label, all_contained_drgus = read_interactions(os.path.join(data_path, "ddi.txt"), smile_graph)
    interactions = interactions_label[:, :2]
    labels = interactions_label[:, 3]

    print("Load PPR network!!")
    all_drug_node = np.unique(interactions.flatten())
    ppr = read_ppr(args.dataset, network_edge_index, num_node, all_drug_node, 0.15, args.eps)

    print("generate subgraphs!!")
    drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(dataset, all_contained_drgus,
                                                                                  network_edge_index, network_rel_index,
                                                                                  num_rel, args)

    data_sta = {
        'num_nodes': num_node,
        'num_rel_mol': num_rel_mol_update + 1,
        'num_rel_graph': num_rel,
        'num_interactions': len(interactions),
        'num_drugs_DDI': len(all_contained_drgus),
        'max_degree_graph': max_smiles_degree + 1,
        'max_degree_node': int(max_subgraph_degree)+1,
        'ppr': ppr
    }

    print(data_sta)

    return interactions, labels, smile_graph, drug_subgraphs, data_sta, network_edge_index, network_rel_index

def save(save_dir, args, train_log, test_log):
    args.device = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f)
    with open(save_dir + '/test_results.json', 'w') as f:
        json.dump(test_log, f)
    with open(save_dir + '/train_log.json', 'w') as f:
        json.dump(train_log, f)

def save_results(save_dir, args, results_list):
    acc = []
    auc = []
    aupr = []
    f1 = []

    for r in results_list:
        acc.append(r['acc'])
        auc.append(r['auc'])
        aupr.append(r['aupr'])
        f1.append(r['f1'])

    acc = np.array(acc)
    auc = np.array(auc)
    aupr = np.array(aupr)
    f1 = np.array(f1)

    results = {
        'acc':[np.mean(acc),np.std(acc)],
        'auc':[np.mean(auc),np.std(auc)],
        'aupr': [np.mean(aupr), np.std(aupr)],
        'f1': [np.mean(f1), np.std(f1)],
    }

    args = vars(args)
    args.update(results)

    with open(save_dir + args['extractor'] + '_all_results.json', 'a+') as f:
        json.dump(args, f)


def init_model(args, dataset_statistics):
    DDI_predictor = Predictor2(max_layer=args.layer,
                    num_features_drug = 67,
                    num_nodes=dataset_statistics['num_nodes'],
                    num_relations_mol=dataset_statistics['num_rel_mol'],
                    num_relations_graph=dataset_statistics['num_rel_graph'],
                    output_dim=args.d_dim,
                    max_degree_graph=dataset_statistics['max_degree_graph'],
                    max_degree_node = dataset_statistics['max_degree_node'],
                    sub_coeff=args.sub_coeff,
                    mi_coeff=args.mi_coeff,
                    dropout=args.dropout,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    ppr=dataset_statistics['ppr'],
                    args=args)
    
    DDI_sampler = Sampler(args, dataset_statistics['num_nodes'])

    predictor_optim = torch.optim.Adam(DDI_predictor.parameters(), args.predictor_lr, weight_decay=args.weight_decay)
    sampler_optim = torch.optim.Adam(DDI_sampler.parameters(), args.sampler_lr, weight_decay=args.weight_decay)

    return DDI_predictor, DDI_sampler, predictor_optim, sampler_optim

def main(args = None, k_fold = 5):

    if args is None:
        args = init_args()

    results_of_each_fold = []

    ##加载interactions的data

    data, labels, smile_graph, node_graph, dataset_statistics, adj_matrix, edge_rel = load_data(args)

    # selected_indices = np.random.choice(data.shape[0], size=data.shape[0]//2, replace=False)
    # data = data[selected_indices]
    # labels = labels[selected_indices]

    edge_index = torch.tensor(adj_matrix).T
    edge_rel = torch.tensor(edge_rel)
    node_graph2 = process_node_graph(data, node_graph, edge_index, edge_rel, args)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_seed(42)
    ##split datasets
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*split_fold(k_fold, data, labels, args, args.s_type))):
        print(f"============================{fold+1}/{k_fold}==================================")
        print("loading data!!")


        ##load_data
        train_data = DTADataset(x=data[train_idx], y=labels[train_idx], sub_graph=node_graph, smile_graph=smile_graph, sub_graph2=node_graph2)
        test_data = DTADataset(x=data[test_idx], y=labels[test_idx], sub_graph=node_graph, smile_graph=smile_graph, sub_graph2=node_graph2)
        eval_data = DTADataset(x=data[val_idx], y=labels[val_idx], sub_graph=node_graph, smile_graph=smile_graph, sub_graph2=node_graph2)
        edge_index = edge_index.cuda()
        edge_rel = edge_rel.cuda()

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate) 
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate) 
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate) 

        DDI_predictor, DDI_sampler, predictor_optim, sampler_optim = init_model(args, dataset_statistics)
        DDI_predictor.to(device)
        DDI_sampler.to(device)
        DDI_predictor.reset_parameters()

        """Load pretrained model"""
        if args.mode == 's1': 
            """Train predictor using trained sampler"""
            # DDI_sampler.load_state_dict(torch.load('/home/xiejc/Code/TIGER/RL/best_save/tiger/drugbank/RL/fold_0/0.00000/DDI_sampler.pt'))
            DDI_sampler.load_state_dict(torch.load(args.load_sampler))
        elif args.mode == 's2': 
            """Train sampler using trained predictor"""
            # DDI_predictor.load_state_dict(torch.load('/home/xiejc/Code/TIGER/RL/best_save/tiger/drugbank/randomWalk/fold_0/0.86878/DDI_predictor.pt')) # 1-hop-RW
            # DDI_predictor.load_state_dict(torch.load('/home/xiejc/Code/TIGER/RL/best_save/tiger/drugbank/khop-subtree/fold_0/0.87003/DDI_predictor.pt')) # 1-hop-k
            DDI_predictor.load_state_dict(torch.load(args.load_predictor))
        elif args.mode == 's3':
            """Train predictor without sampler"""
            pass
        elif args.mode == 's4':
            """Train both predictor and sampler"""
            pass
        

        ##train_model
        #trange = tqdm(range(1, args.epoch + 1))
        best_auc = 0.0
        early_stop_num = 0
        train_reward = 0

        train_log = {'train_acc':[], 'train_auc':[], 'train_aupr':[], 'train_loss':[], 'train_reward':[],
                        'eval_f1':[], 'eval_acc':[], 'eval_auc':[], 'eval_aupr':[], 'eval_loss':[], 'eval_rewrad':[]}
        
        for i_episode in range(args.epoch):
            loop = tqdm(train_loader, ncols=80)
            loop.set_description(f'Epoch[{i_episode}/{args.epoch}]')
            train_acc, train_f1, train_auc, train_aupr, train_loss, train_reward = train(
                loop, 
                DDI_predictor, 
                DDI_sampler, 
                predictor_optim, 
                sampler_optim, 
                edge_index, 
                edge_rel,
                train_reward, 
                args,
                dataset_statistics
            )

            eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss, eval_rewrad = eval(test_loader, DDI_predictor, DDI_sampler, edge_index, edge_rel, dataset_statistics, args)
            print(f"train_auc:{train_auc} train_aupr:{train_aupr} train_reward: {train_reward} eval_auc:{eval_auc} eval_aupr:{eval_aupr}, eval_reward: {eval_rewrad}")

            train_log['train_acc'].append(train_acc)
            train_log['train_auc'].append(train_auc)
            train_log['train_aupr'].append(train_aupr)
            train_log['train_loss'].append(train_loss)
            train_log['train_reward'].append(train_reward)

            train_log['eval_acc'].append(eval_acc)
            train_log['eval_f1'].append(eval_f1)
            train_log['eval_auc'].append(eval_auc)
            train_log['eval_aupr'].append(eval_aupr)
            train_log['eval_loss'].append(eval_loss)
            train_log['eval_rewrad'].append(eval_rewrad)

            best_model_state11 = copy.deepcopy(DDI_predictor.state_dict())
            best_model_state22 = copy.deepcopy(DDI_sampler.state_dict())
            if args.mode == 's2':
                save_dir = os.path.join('./best_save/', args.model_name, args.dataset, args.extractor, '{}-{}-{}'.format(args.neg, args.pos, args.mode),
                                        "fold_{}".format(fold), "{:.5f}".format(train_reward))
            elif args.mode == 's1':
                save_dir = os.path.join('./best_save/', args.model_name, args.dataset, args.extractor, args.mode,
                                        "fold_{}".format(fold), "{:.5f}".format(eval_auc))
            else:
                save_dir = None
            
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(best_model_state11, os.path.join(save_dir, 'DDI_predictor.pt'))
                torch.save(best_model_state22, os.path.join(save_dir, 'DDI_sampler.pt'))

            if eval_auc > best_auc:
                best_model_state1 = copy.deepcopy(DDI_predictor.state_dict())
                best_model_state2 = copy.deepcopy(DDI_sampler.state_dict())
                best_auc = eval_auc
                early_stop_num = 0
            else:
                early_stop_num += 1
                if early_stop_num > 10:
                    print("early stop!")
                    break

        # DDI_predictor.load_state_dict(best_model_state1)
        # DDI_sampler.load_state_dict(best_model_state2)
        # model.to(device)
        # test_log = test(test_loader, model) ##test_log是一个字典，里面存储着metrics

        best_epoch = np.argmax(np.array(train_log['eval_auc']))
        print('Best epoch: %d' % best_epoch)
        print('acc: %.4f' % train_log['eval_acc'][best_epoch])
        print('f1: %.4f' % train_log['eval_f1'][best_epoch])
        print('auc: %.4f' % train_log['eval_auc'][best_epoch])
        print('aupr: %.4f' % train_log['eval_aupr'][best_epoch])


        if args.mode == 's2':
            save_dir = os.path.join('./best_save/', args.model_name, args.dataset, args.extractor, '{}-{}-{}'.format(args.neg, args.pos, args.mode),
                                    "fold_{}".format(fold), "best")
        else:
            save_dir = os.path.join('./best_save/', args.model_name, args.dataset, args.extractor, args.mode,
                                    "fold_{}".format(fold), "{:.5f}".format(best_auc))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(best_model_state1, os.path.join(save_dir, 'DDI_predictor.pt'))
        torch.save(best_model_state2, os.path.join(save_dir, 'DDI_sampler.pt'))
        test_log = {}
        save(save_dir, args, train_log, test_log)
        print(f"save to {save_dir}")
        # results_of_each_fold.append(test_log)


    save_results(os.path.join('./best_save/', args.model_name, args.dataset), args, results_of_each_fold)

    return


if __name__ == "__main__":
    main()
