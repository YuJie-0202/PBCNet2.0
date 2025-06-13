import torch
import torch.nn as nn
import pickle

from torch import Tensor


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_count_gradient(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if torch.isnan(param) is True)


def pkl_load(file_name):

    pickle_file = open(file_name, 'rb')
    graph = pickle.load(pickle_file)
    return graph


def pkl_load_no_AR(file_name):
    pickle_file = open(file_name, 'rb')
    graph = pickle.load(pickle_file)
    graph.remove_nodes(torch.where(torch.sum(graph.ndata['atom_feature'],dim=1)==0 )[0])
    return graph


def pkl_load_gm(file_name):
    pickle_file = open(file_name, 'rb')
    a = pickle.load(pickle_file)
    return a


def Extend(list1, list2):
    list1.extend(list2)
    return list1


def gm_to_batch(gm, lenth):
    for x in range(len(gm)):
        if x == 0:
            continue
        gm[x] = gm[x] + lenth[x - 1]

    gm = torch.cat(gm, dim=0)
    return gm


def gm_process(path_list, graph, device):
    dist = []
    angle = []
    torsion = []
    i = []
    j = []
    idx_kj = []
    idx_ji = []
    incomebond_edge_ids = []
    incomebond_index_to_atom = []
    for s in path_list:
        dist_, angle_, torsion_, i_, j_, idx_kj_, idx_ji_, incomebond_edge_ids_, incomebond_index_to_atom_ = pkl_load_gm(
            s.rsplit(".", 1)[0] + "_gm.pkl")
        dist.append(dist_)
        angle.append(angle_)
        torsion.append(torsion_)
        i.append(i_)
        j.append(j_)
        idx_kj.append(idx_kj_)
        idx_ji.append(idx_ji_)
        incomebond_edge_ids.append(incomebond_edge_ids_)
        incomebond_index_to_atom.append(incomebond_index_to_atom_)
    dist = torch.cat(dist, dim=0)
    angle = torch.cat(angle, dim=0)
    torsion = torch.cat(torsion, dim=0)

    n_nodes = graph.batch_num_nodes()
    n_nodes = torch.tensor([torch.sum(n_nodes[0:i + 1]) for i in range(len(n_nodes))])

    n_edges = graph.batch_num_edges()
    n_edges = torch.tensor([torch.sum(n_edges[0:i + 1]) for i in range(len(n_edges))])

    i = gm_to_batch(i, n_nodes)
    j = gm_to_batch(j, n_nodes)
    idx_kj = gm_to_batch(idx_kj, n_edges)
    idx_ji = gm_to_batch(idx_ji, n_edges)
    incomebond_edge_ids = gm_to_batch(incomebond_edge_ids, n_edges)
    incomebond_index_to_atom = gm_to_batch(incomebond_index_to_atom, n_nodes)

    return dist.to(device), angle.to(device), torsion.to(device), i.to(device), j.to(device), idx_kj.to(device), \
           idx_ji.to(device), incomebond_edge_ids.to(device), incomebond_index_to_atom.to(device)
