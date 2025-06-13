from rdkit.Chem.rdchem import BondType as BT
import pickle
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
import pandas as pd
import torch
import os
import warnings
import multiprocessing
import dgl
from Bio.PDB.PDBParser import PDBParser

def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

warnings.filterwarnings('ignore')

allowable_features = {
    # 'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': ['CHI_UNSPECIFIED',
                                'CHI_TETRAHEDRAL_CW',
                                'CHI_TETRAHEDRAL_CCW',
                                'CHI_OTHER'],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True]
    }

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1

# residue Information , not used in PBCNet2.0!
# ====== for pockt group =======
one_to_three = {"A" : "ALA",
              "C" : "CYS",
              "D" : "ASP",
              "E" : "GLU",
              "F" : "PHE",
              "G" : "GLY",
              "H" : "HIS",
              "I" : "ILE",
              "K" : "LYS",
              "L" : "LEU",
              "M" : "MET",
              "N" : "ASN",
              "P" : "PRO",
              "Q" : "GLN",
              "R" : "ARG",
              "S" : "SER",
              "T" : "THR",
              "V" : "VAL",
              "W" : "TRP",
              "Y" : "TYR",
              "B" : "ASX",
              "Z" : "GLX",
              "X" : "UNK",
              "*" : " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

def prop(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                        1 if residue in pro_res_polar_neutral_table else 0,
                        1 if residue in pro_res_acidic_charged_table else 0,
                        1 if residue in pro_res_basic_charged_table else 0, 1]
    return res_property1

def res_pocket(p):

    parser = PDBParser(PERMISSIVE=1)
    s = parser.get_structure('a', p)
   
    atom2res_idx = []
    res_name = []
    count = -1
    for model in s:
        for chain in model:
            for residue in chain:
                count += 1
                name = residue.get_resname()
                for atom in residue:
                    if atom.get_fullname() [0] != 'H' and atom.get_fullname()[1] != 'H':
                        atom2res_idx.append(count)
                        res_name.append(name)

    res3 = [three_to_one[i] if i in three_to_one.keys() else 'X' for i in res_name]

    res_type = [pro_res_table.index(i) for i in res3]  
    res_prop  = [prop(i) for i in res3] 
    assert len(atom2res_idx) == len(res_type) == len(res_prop)
    return atom2res_idx, res_type, res_prop


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            # allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            # allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            # allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            # allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            # allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            # allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return atom_features_list


def group_complex(ligand, pocket_dir):
    brics_index = brics_decomp(ligand)
    group_index = [0 for _ in range(len(ligand.GetAtoms()))]
    group_type = [0 for _ in range(len(ligand.GetAtoms()))]   # 0
    group_prop = [[0,0,0,0,0,0] for _ in range(len(ligand.GetAtoms()))] # 0

    if type(brics_index) == type(tuple(('a', 'b', 'c'))):
        brics_index = brics_index[0]

    for i, idx in enumerate(brics_index):
        for idx_ in idx:
            group_index[idx_] = i

    atom2res_idx, res_type, res_prop = res_pocket(pocket_dir)

    atom2res_idx = [i+len(brics_index) for i in atom2res_idx] 
    res_type = [i+1 for i in res_type] 

    group_index.extend(atom2res_idx)
    group_type.extend(res_type)
    group_prop.extend(res_prop)

    return torch.tensor(group_index,dtype=torch.float32), torch.tensor(group_type,dtype=torch.float32), torch.tensor(group_prop, dtype=torch.float32)


def atom_type(ligand,pocket):
    # ====== 获得原子类型 ======
    l_atom = np.array([i.GetAtomicNum() for i in ligand.GetAtoms()])
    lig_atom = [1 for _ in l_atom]
    p_atom = np.array([i.GetAtomicNum() for i in pocket.GetAtoms()])
    pock_atom = [0 for _ in p_atom]
    complex_atom = torch.tensor(np.concatenate([l_atom, p_atom]))
    type_for_mask = torch.tensor(np.concatenate([lig_atom, pock_atom]))
    z = complex_atom
    return z, type_for_mask


bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} # 4:分子内部边; 5:蛋白内部边; 6:蛋白-配体边

def bond_featurizer(ligand, pocket, index):
    bond_type = []
    num_atoms = len(ligand.GetAtoms())
    for a1, a2 in zip(index[0], index[1]):
        if a1 < num_atoms and a2 < num_atoms:   # in ligand
            bond = ligand.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                bond_type.append(4)   # distance bond
            else:
                bond_type.append(bonds[bond.GetBondType()])
        elif a1 >= num_atoms and a2 >= num_atoms: # in pock
            a1 = a1 - num_atoms
            a2 = a2 - num_atoms
            bond = pocket.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                bond_type.append(4)   # distance bond
            else:
                bond_type.append(bonds[bond.GetBondType()])
        else:
            bond_type.append(4)
    return torch.tensor(bond_type, dtype=torch.float32)
    

# ===== mol split =====
# R-group Information , not used in PBCNet2.0!
def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques


def Graph_Information(ligand_file, pocket_file):

    ligand = Chem.MolFromMolFile(ligand_file)
    ligand = Chem.RemoveAllHs(ligand)
    pocket = Chem.MolFromPDBFile(pocket_file)
    pocket = Chem.RemoveAllHs(pocket)


    graph_data = {
                ('atom', 'int', 'atom'): ([], []),
                ('atom', 'ind', 'atom'): ([], [])      # 'ind' bond is not used in PBCNet2.0

                }
    G = dgl.heterograph(graph_data)

    # ====== atom type =======
    x, type_for_mask = atom_type(ligand, pocket)
    G.add_nodes(x.shape[0])
    
    # ====== chemical information =======
    lig_atom_feature = lig_atom_featurizer(ligand)
    pock_atom_feature = lig_atom_featurizer(pocket)
    atom_scalar = torch.tensor(np.concatenate([lig_atom_feature, pock_atom_feature]), dtype=torch.float32)

    # ===== group infor =====
    index, g_type, g_prop = group_complex(ligand, pocket_file)

    # ====== coor ======
    coor_lig = ligand.GetConformer().GetPositions()
    coor_pock = pocket.GetConformer().GetPositions()
    pos = np.concatenate([coor_lig, coor_pock])
    pos = torch.tensor(pos, dtype=torch.float32)
    G.nodes['atom'].data['pos'] = pos
    
    # ====== type1: in mol (all: dist and or) ======  
    for i in range(len(coor_lig)):
        for j in range(i + 1, len(coor_lig)):
            dist = np.linalg.norm(coor_lig[i] - coor_lig[j])
            if dist <= 5 and dist > 0:
                G.add_edges(i, j, etype='int')
                G.add_edges(j, i, etype='int')
                continue

    # ====== type2: in pock (co) ======  
    for bond in pocket.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        G.add_edges(start_atom + len(coor_lig), end_atom + len(coor_lig), etype='int')
        G.add_edges(end_atom + len(coor_lig), start_atom + len(coor_lig), etype='int')
        continue


    # ====== type3: in ligand and pock  ======  
    for i in range(len(coor_lig)):
        for j in range(len(coor_pock)):
            dist = np.linalg.norm(coor_lig[i] - coor_pock[j])
            if dist <= 5 and dist > 0:
                G.add_edges(i, len(coor_lig) + j, etype='int')
                G.add_edges(len(coor_lig) + j, i, etype='int')
                continue
    
    # ====== type4: in pock (distance) ======  
    connected_protein_atoms = set(j for i in range(len(coor_lig)) for j in G.successors(i,etype='int') if j >= len(coor_lig))
    # print(connected_protein_atoms)
    for i in connected_protein_atoms:
        for j in range(len(coor_pock)):
            dist = np.linalg.norm(coor_pock[i - len(coor_lig)] - coor_pock[j])
            if  (dist <= 3) and (dist > 0) and  (G.has_edges_between(i, len(coor_lig) + j, etype='int').item() is False):
                G.add_edges(i, len(coor_lig) + j, etype='int')
                G.add_edges(len(coor_lig) + j, i, etype='int')
                continue

    edge_index_int = [G.edges(etype='int')[0].detach().numpy().tolist(), G.edges(etype='int')[1].detach().numpy().tolist()]
    edge_index_ind = [G.edges(etype='ind')[0].detach().numpy().tolist(), G.edges(etype='ind')[1].detach().numpy().tolist()]

    bond_type_int = bond_featurizer(ligand, pocket, edge_index_int)
    bond_type_ind = bond_featurizer(ligand, pocket, edge_index_ind)
    
    #index, g_type, g_prop
    G.nodes['atom'].data['res_idx'] = index
    G.nodes['atom'].data['res_type'] = g_type
    G.nodes['atom'].data['res_prop'] = g_prop

    G.nodes['atom'].data['x'] = x
    G.nodes['atom'].data['pos'] = pos
    G.nodes['atom'].data['type'] =type_for_mask
    G.nodes['atom'].data['atom_scalar'] = atom_scalar
    G.edges['ind'].data['bond_scalar'] = bond_type_ind
    G.edges['int'].data['bond_scalar'] = bond_type_int
    return G


def graph_save(ligand_file, pock_file, pickle_save):
    
    if not (os.path.exists(ligand_file) and os.path.exists(pock_file)):
        return None

    data = Graph_Information(ligand_file, pock_file)
    pickle_save = open(pickle_save, 'wb')
    pickle.dump(data, pickle_save)
    pickle_save.close()

def parallel(z):
    return graph_save(z[0], z[1], z[2])

def print_error(value):
    print("error: ", value)


if __name__ == "__main__":

    setup_cpu(1)
    data_list = []

    sys = ['Jnk1']
    fep = [f'/home/user-home/yujie/0_PBCNetv2/data/FEP/{i}/pose/' for i in sys]

    for s in fep:
        ligand_file = [s +'/'+ a for a in os.listdir(s) if a.endswith('.sdf')]
        pock_file = [c.rsplit('/', 1)[0] + '/pocket.pdb' for c in ligand_file]
        pickle_file = [c.replace('.sdf', '_dgl_group.pkl') for c in ligand_file]
        
        for g in range(len(ligand_file)):
            data_list.append([ligand_file[g], pock_file[g], pickle_file[g]])
    print(len(data_list))


    pool = multiprocessing.Pool(50)
    for i in data_list:
        pool.apply_async(graph_save, i, error_callback=print_error)
    pool.close()
    pool.join()