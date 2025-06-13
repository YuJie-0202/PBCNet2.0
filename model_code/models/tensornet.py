import torch
from typing import Optional, Tuple
from torch import Tensor, nn
import dgl
import dgl.function as fn
import sys
import os
code_path =  os.path.dirname(os.path.abspath(__file__))    # /home/user-home/yujie/0_PBCNetv2/0_PBCNET/model_code/train.py/
code_path = code_path.rsplit("/", 1)[0]
sys.path.append(code_path)
from models.utils import (CosineCutoff,
                          rbf_class_mapping,
                          act_class_mapping,)

__all__ = ["TensorNet"]
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S


def tensor_norm(tensor):
    """Computes Frobenius norm."""
    return (tensor**2).sum((-2, -1))


class TensorNet(nn.Module):

    def __init__(
        self,
        hidden_channels=128,
        num_layers=5,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_lower=0,
        cutoff_upper=5,
        max_z=128,
        equivariance_invariance_group="O(3)",
        dtype=torch.float32,
    ):
        super(TensorNet, self).__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert equivariance_invariance_group in ["O(3)", "SO(3)"], (
            f'Unknown group "{equivariance_invariance_group}". '
            f"Choose O(3) or SO(3)."
        )
        
        self.hidden_channels = hidden_channels
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        act_class = act_class_mapping[activation]
        # rbf
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff_lower,
                                                              cutoff_upper,
                                                              num_rbf,
                                                              trainable_rbf)
        # for bond type: bond chemical informaton
        self.bond_emb = torch.nn.Embedding(5, self.num_rbf)
        # Initial tensor representations
        self.tensor_embedding = TensorEmbedding(hidden_channels,
                                                num_rbf,
                                                act_class,
                                                cutoff_lower,
                                                cutoff_upper,
                                                trainable_rbf,
                                                max_z,
                                                dtype)

        self.layers = nn.ModuleList()
        if num_layers != 0:
            for _ in range(num_layers):
                self.layers.append(Interaction(num_rbf,
                                               hidden_channels,
                                               act_class,
                                               cutoff_lower,
                                               cutoff_upper,
                                               equivariance_invariance_group,
                                               dtype))
        self.linear = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)
        self.out_norm = nn.LayerNorm(3 * hidden_channels, dtype=dtype)
        self.act = act_class()

        self.reset_parameters()

    def reset_parameters(self):
        # self.tensor_embedding.reset_parameters()
        # for layer in self.layers:
        #     layer.reset_parameters()
        self.linear.reset_parameters()
        self.out_norm.reset_parameters()
        self.bond_emb.reset_parameters()

    def forward(self,
                g, bb=False) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        
        with g.local_scope():
            
            self.dtype = torch.float32
            self.device = g.device
            # Obtain rbf embedding, bond type embedding and relative position vectors
            g.nodes['atom'].data['x'] = g.nodes['atom'].data['x'].long()
            g.edges['int'].data['rbf'] = torch.cat((self.distance_expansion(g.edges['int'].data['dist']), self.bond_emb(g.edges['int'].data['bond_scalar'].long())),dim=-1)
            g.edges['int'].data['vec_norm'] = g.edges['int'].data['vec'] / g.edges['int'].data['dist'].unsqueeze(1)
            
            # Initial tensor representations
            X = self.tensor_embedding(g, bond_type = 'int')

            # Tensor representations updating
            ATT = []
            for num_layer_ in range(len(self.layers)):
                layer = self.layers[num_layer_]
                X, att = layer(g, X, bond_type = 'int')
                ATT.append(att)

            I, A, S = decompose_tensor(X)
            x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
            x = self.out_norm(x)
            x = self.act(self.linear((x)))
            return x, ATT



def src_cat_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.cat((edges.src[src_field], edges.dst[dst_field]), dim=-1)}
    return func

def e_cat_e(edge_field1, edge_field2, out_field):
    def func(edges):
        return {out_field: torch.cat((edges.data[edge_field1], edges.data[edge_field2]), dim=-1)}
    return func


def e_mul_e(edge_field1, edge_field2, out_field):
    def func(edges):
        # clamp for softmax numerical stability
        return {out_field: edges.data[edge_field1] * edges.data[edge_field2]}
    return func

def src_dst_e_cat(src_field, dst_field, edge_field, out_field):
    def func(edges):
        return {out_field: torch.cat((edges.src[src_field], edges.dst[dst_field], edges.data[edge_field]), dim=-1)}
    return func


class AtomEncoder(torch.nn.Module):

    # to encode the atom chemical features

    def __init__(self, 
                 feature_dims, 
                 hidden_dim):
        # feature_dims a list with the lenght of each categorical feature, here : [10,20,10,10,20,20,10,10,10]

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)  # 7

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, hidden_dim, dtype=torch.float32)
            self.atom_embedding_list.append(emb)
        
        self.reset_parameters()

    def reset_parameters(self):
        for emb_ in self.atom_embedding_list:
            emb_.reset_parameters()

    def forward(self,
                x):
        # x : chemical features of atoms
        x_embedding = 0
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
        return x_embedding



class TensorEmbedding(nn.Module):
    """
       Initial tensor representations: to get X_i of each atom. (Done)
    """

    def __init__(self,
                 hidden_channels,
                 num_rbf,
                 activation,
                 cutoff_lower,
                 cutoff_upper,
                 trainable_rbf=False,
                 max_z=128,
                 dtype=torch.float32,
                 ):
        super(TensorEmbedding, self).__init__()

        self.num_rbf = num_rbf*2
        num_rbf = num_rbf*2

        self.hidden_channels = hidden_channels
        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = nn.Embedding(max_z, hidden_channels, dtype=dtype)              # atom type
        self.AE = AtomEncoder([10,20,10,10,20,20,10,10,10],self.hidden_channels)  # other atomic chemical features
        self.emb2 = nn.Linear(2 * hidden_channels, hidden_channels, dtype=dtype)
        self.act = activation()
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=dtype)

        self.linears_tensor = nn.ModuleList()
        for _ in range(3):
            self.linears_tensor.append( nn.Linear(hidden_channels, hidden_channels, bias=False) )

        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append( nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype) )
        self.linears_scalar.append( nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype) )
        
        self.reset_parameters()

    def reset_parameters(self):
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def _get_atomic_number_message(self,
                                   g,
                                   bond_type):
        with g.local_scope():
            g.nodes['atom'].data['Z'] = self.emb(g.nodes['atom'].data['x']) + self.AE(g.nodes['atom'].data['atom_scalar'])
            g.apply_edges(src_cat_dst('Z', 'Z', 'Zij'), etype=bond_type)
            Zij = self.emb2(g.edges[bond_type].data['Zij'])[..., None, None]   # shape: edges * (2*hidden dim) --> edges * hidden dim * 1 * 1
            return Zij

    def _get_tensor_messages(self, 
                             g,
                             bond_type,
                             Zij):
        with g.local_scope():
            g.edges[bond_type].data['Zij'] = Zij
            g.edges[bond_type].data['cut'] = self.cutoff(g.edges[bond_type].data['dist']).reshape(-1, 1, 1, 1)     # Deprecated
            g.edges[bond_type].data['cut'] = torch.ones(g.edges[bond_type].data['cut'].shape, device=g.edges[bond_type].data['cut'].device)  # to cover 'g.edges[bond_type].data['cut']'
            g.apply_edges(e_mul_e('cut', 'Zij', 'C'), etype=bond_type)

            eye = torch.eye(3, 3, device=g.device, dtype=torch.float32)[None, None, ...]
            g.edges[bond_type].data['Iij'] = self.distance_proj1(g.edges[bond_type].data['rbf'])[..., None, None]
            g.apply_edges(e_mul_e('Iij', 'C', 'Iij'), etype=bond_type)
            Iij = g.edges[bond_type].data['Iij'] * eye

            g.edges[bond_type].data['Aij'] = self.distance_proj2(g.edges[bond_type].data['rbf'])[..., None, None]
            g.apply_edges(e_mul_e('Aij', 'C', 'Aij'), etype=bond_type)
            Aij = g.edges[bond_type].data['Aij'] * vector_to_skewtensor(g.edges[bond_type].data['vec_norm'])[..., None, :, :]

            g.edges[bond_type].data['Sij'] = self.distance_proj3(g.edges[bond_type].data['rbf'])[..., None, None]
            g.apply_edges(e_mul_e('Sij', 'C', 'Sij'), etype=bond_type)
            Sij = g.edges[bond_type].data['Sij'] * vector_to_symtensor(g.edges[bond_type].data['vec_norm'])[..., None, :, :]
            
            return Iij, Aij, Sij

    def forward(self,
                g,
                bond_type ):
        with g.local_scope():
            Zij = self._get_atomic_number_message(g, bond_type)
            g.edges[bond_type].data['Iij'], g.edges[bond_type].data['Aij'], g.edges[bond_type].data['Sij']= self._get_tensor_messages(g, bond_type, Zij)

            g.update_all(fn.copy_e('Iij', 'm'), fn.sum('m', 'I'), etype=bond_type)
            g.update_all(fn.copy_e('Aij', 'm'), fn.sum('m', 'A'), etype=bond_type)
            g.update_all(fn.copy_e('Sij', 'm'), fn.sum('m', 'S'), etype=bond_type)

            I, A, S = g.nodes['atom'].data['I'], g.nodes['atom'].data['A'], g.nodes['atom'].data['S']

            norm = self.init_norm(tensor_norm(I + A + S))

            for linear_scalar in self.linears_scalar:
                norm = self.act(linear_scalar(norm))

            norm = norm.reshape(-1, self.hidden_channels, 3)
            I = (self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                 *norm[..., 0, None, None])
            A = (self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                 *norm[..., 1, None, None])
            S = (self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                *norm[..., 2, None, None])
                
            X = I + A + S
            return X


def tensor_message_passing( g, 
                            factor: Tensor, 
                            tensor: Tensor, 
                            bond_type       ) -> Tensor:
    """
    Message passing for tensors.
    g:graph
    factor: edge_attr
    tensor: I, S, A
    """
    with g.local_scope():
        g.edges[bond_type].data['F'] = factor
        g.nodes['atom'].data['T'] = tensor
        g.update_all(fn.u_mul_e('T','F', 'm'), fn.sum('m', 'tensor_m'), etype=bond_type)
        return g.nodes['atom'].data['tensor_m']


class Interaction(nn.Module):
    """Interaction layer.
       Tensor representations updating
    """

    def __init__(self,
                 num_rbf,
                 hidden_channels,
                 activation,
                 cutoff_lower,
                 cutoff_upper,
                 equivariance_invariance_group,
                 dtype=torch.float32
                ):
        
        super(Interaction, self).__init__()

        self.atom_inf = True

        if self.atom_inf is True:
            self.num_rbf = (num_rbf + hidden_channels) * 2 
            num_rbf = (num_rbf + hidden_channels) * 2 
        else:
            self.num_rbf = num_rbf*2 
            num_rbf = num_rbf*2

        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)   # Deprecated
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append( nn.Linear(num_rbf, hidden_channels, bias=True, dtype=dtype)  )
        self.linears_scalar.append( nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype) )
        self.linears_scalar.append( nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype) )

        self.linears_tensor = nn.ModuleList()
        for _ in range(6):
            self.linears_tensor.append( nn.Linear(hidden_channels, hidden_channels, bias=False) )
        
        self.act = activation()
        self.equivariance_invariance_group = equivariance_invariance_group

        if self.atom_inf is True:
            self.out_norm_ = nn.LayerNorm(3 * hidden_channels, dtype=dtype)
            self.linear_ = nn.Linear(3*hidden_channels, hidden_channels, dtype=dtype)
        
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        
        if self.atom_inf is True:
            self.out_norm_.reset_parameters()
            self.linear_.reset_parameters()

    def forward(self,
                g,
                X: Tensor,
                bond_type
               ):
        with g.local_scope():
            
            if self.atom_inf is True:
                # Concatenate the invariant features of the two connected atoms onto the invariant features of the edge.
                # Scalar embedding extracting
                I, A, S = decompose_tensor(X)
                x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
                x = self.out_norm_(x)
                x = self.act(self.linear_(x))
                g.nodes['atom'].data['att_atom'] = x

                g.apply_edges(src_dst_e_cat('att_atom','att_atom', 'rbf', 'rbf'), etype=bond_type)
                edge_attr = g.edges[bond_type].data['rbf']
            else:
                edge_attr = g.edges[bond_type].data['rbf']
            
            edge_weight = g.edges[bond_type].data['dist']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            C = self.cutoff(edge_weight)  # Deprecated
            C = torch.ones(C.shape, device=C.device)  # to cover "C"
            
            for linear_scalar in self.linears_scalar:
                edge_attr = self.act(linear_scalar(edge_attr))

            edge_attr = (edge_attr * C.view(-1, 1)).reshape(
                edge_attr.shape[0], self.hidden_channels, 3
            )
            
            X = X / (tensor_norm(X) + 1)[..., None, None]
            I, A, S = decompose_tensor(X)
            I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            Y = I + A + S
            Im = tensor_message_passing(
                g, edge_attr[..., 0, None, None], I, bond_type
            )
            Am = tensor_message_passing(
                g, edge_attr[..., 1, None, None], A, bond_type
            )
            Sm = tensor_message_passing(
                g, edge_attr[..., 2, None, None], S, bond_type
            )
            
            msg = Im + Am + Sm
            
            if self.equivariance_invariance_group == "O(3)":
                A = torch.matmul(msg, Y)
                B = torch.matmul(Y, msg)
                I, A, S = decompose_tensor(A + B)
            if self.equivariance_invariance_group == "SO(3)":
                B = torch.matmul(Y, msg)
                I, A, S = decompose_tensor(2 * B)
            normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
            I, A, S = I / normp1, A / normp1, S / normp1
            I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            dX = I + A + S
            X = X + dX + torch.matrix_power(dX, 2)
            return X, edge_attr
