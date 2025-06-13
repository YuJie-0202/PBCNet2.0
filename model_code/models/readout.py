import torch
from torch import nn
import dgl
import sys
import os
code_path =  os.path.dirname(os.path.abspath(__file__))    # /home/user-home/yujie/0_PBCNetv2/0_PBCNET/model_code/train.py/
code_path = code_path.rsplit("/", 1)[0]
sys.path.append(code_path)
from models.utils import act_class_mapping
from models.tensornet import TensorNet
import dgl.function as fn

def src_norm_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.sqrt(torch.sum((edges.src[src_field] - edges.dst[dst_field]) ** 2, dim=1))}
    return func

class PBCNetv2(nn.Module):
    def __init__(self, 
                 hidden_channels = 128,
                 num_layer = 5,
                 num_rbf = 32,
                 max_z = 128,
                 equivariance_invariance_group = "O(3)",
                 activation = 'silu',
                 dtype=torch.float32):
        super(PBCNetv2, self).__init__()
        
        self.hidden_channels = hidden_channels

        self.encoder = TensorNet(hidden_channels = self.hidden_channels,
                                num_layers = num_layer,
                                num_rbf = num_rbf,
                                rbf_type = "expnorm",
                                trainable_rbf = False,
                                activation = activation,
                                cutoff_lower = 0,
                                cutoff_upper = 5,
                                max_z = max_z,
                                equivariance_invariance_group = equivariance_invariance_group,
                                dtype=torch.float32)


        act_class = act_class_mapping[activation]
        self.act = act_class()
        
        self.norm = nn.LayerNorm(self.hidden_channels , dtype=dtype)
        ffn = [nn.Linear(self.hidden_channels, self.hidden_channels * 2,dtype=dtype), self.act]
        ffn.extend([nn.Linear(self.hidden_channels * 2, self.hidden_channels, dtype=dtype), self.act])
        ffn.extend([nn.Linear(self.hidden_channels, self.hidden_channels // 2, dtype=dtype), self.act])
        ffn.append(nn.Linear(self.hidden_channels // 2, 1,dtype=dtype))   
        
        self.FNN = nn.Sequential(*ffn)
        self.reset_parameters()

    def reset_parameters(self):
        count = -1
        for layer in self.FNN:
            count += 1
            if count in [1,3,5]:
                continue
            layer.reset_parameters()
        self.norm.reset_parameters()
        
    def _readout(self,
                 g,
                 bb):
        with g.local_scope():
            # encode and readout
            g.apply_edges(src_norm_dst('pos', 'pos', 'dist'), etype='int')
            g.apply_edges(fn.v_sub_u('pos', 'pos', 'vec'), etype='int')

            emb,att = self.encoder(g, bb)
            mask = g.nodes['atom'].data['type'].unsqueeze(1).repeat(1,self.hidden_channels)
            g.nodes['atom'].data['emb'] = emb * mask  # mask the protein atoms
            # reduce
            emb_mol = dgl.readout_nodes(g, 'emb', ntype='atom')
            return emb_mol,att
    
    def forward(self,
                g1,
                g2,
                bb=False):

        g1.edges['int'].data['bond_scalar'] = torch.clip(g1.edges['int'].data['bond_scalar'].long(), min=0, max=4)   # 5 type of bond
        g2.edges['int'].data['bond_scalar'] = torch.clip(g2.edges['int'].data['bond_scalar'].long(), min=0, max=4)

        emb1,att1 = self._readout(g1, bb) # [batch, hidden_dim]
        emb2,att2 = self._readout(g2, bb)

        emb = self.norm(emb1-emb2)
        out_put = self.FNN(emb)

        emb_neg = self.norm(emb2-emb1)
        out_put_neg = self.FNN(emb_neg)

        return out_put, out_put_neg


        