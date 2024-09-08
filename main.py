import os
import numpy as np

import time
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from Bio.PDB import PDBParser
import torch
from torch.optim import Adam
from model import GraphVAE


ATOM_TYPE_MAPPING = [
    "A_P", "A_O5'", "A_C5'", "A_C4'", "A_O4'", "A_C3'", "A_O3'", "A_C2'", "A_O2'", "A_C1'",
    "A_N9", "A_C8", "A_N7", "A_C5", "A_C6", "A_N6", "A_N1", "A_C2", "A_N3", "A_C4",
    "C_P", "C_O5'", "C_C5'", "C_C4'", "C_O4'", "C_C3'", "C_O3'", "C_C2'", "C_O2'", "C_C1'",
    "C_N1", "C_C2", "C_O2", "C_N3", "C_C4", "C_N4", "C_C5", "C_C6",
    "G_P", "G_O5'", "G_C5'", "G_C4'", "G_O4'", "G_C3'", "G_O3'", "G_C2'", "G_O2'", "G_C1'",
    "G_N9", "G_C8", "G_N7", "G_C5", "G_C6", "G_O6", "G_N1", "G_C2", "G_N2", "G_N3", "G_C4",
    "U_P", "U_O5'", "U_C5'", "U_C4'", "U_O4'", "U_C3'", "U_O3'", "U_C2'", "U_O2'", "U_C1'",
    "U_N1", "U_C2", "U_O2", "U_N3", "U_C4", "U_O4", "U_C5", "U_C6",
    "U_H5T", "U_H3T", "U_HO3'", "U_HO5'", "U_O5''", "U_O3''", "U_OP1", "U_OP2", "U_O2''",
    "U_H2'", "U_H3'", "U_H4'", "U_H5'", "U_H5''", "U_H1'", "U_H2''", "G_OP1", "G_OP2", "C_OP1", "C_OP2", "A_OP1", "A_OP2", "G_OP3", "C_OP3", "U_OP3", "A_OP3"
]


def is_hetero_atom(atom):
    """
    Check if the given atom is a hetero atom.
    """
    residue = atom.get_parent()
    het_flag, resseq, icode = residue.id
    return het_flag != ' '


def calculate_distance_matrix(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)
    model = structure[0]
    # Extract all CA (alpha carbon) atoms (or any other atoms of interest)
    atoms = [atom for atom in model.get_atoms() if atom.get_name()[:1] != 'H' and not is_hetero_atom(atom)]

    # Extract coordinates
    coords = np.array([atom.get_coord() for atom in atoms if atom.get_name()[:1] != 'H' and not is_hetero_atom(atom)])

    # Calculate pairwise distances
    num_atoms = len(coords)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])
            if (distance < 3.5):
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric matrix

    return distance_matrix, atoms

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_atom_type(atom):
    """
    Given an atom, return the atom type in the format 'Residue_AtomName'.
    """
    residue = atom.get_parent()
    residue_name = residue.get_resname()
    if residue_name in ['DA', 'DC', 'DG', 'DU']:
        residue_name = residue_name[1]
    if residue_name not in ['A', 'C', 'G', 'U'] and not is_hetero_atom(atom):
        return None
    atom_name = atom.get_name()
    atom_type = f"{residue_name}_{atom_name}"
    return atom_type

def create_graph_from_distance_matrix(distance_matrix, atoms):
    edges = np.transpose(np.nonzero(distance_matrix))
    weights = distance_matrix[edges[:, 0], edges[:, 1]]
    
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float)
    
    num_nodes = distance_matrix.shape[0]
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    # Encode node features (atom types) as numerical values
    types = [get_atom_type(atom) for atom in atoms]
    if (len(atoms) > 10000):
        return None
    if any(element is None for element in types):
        return None
    print("it is good structure")
    node_features = np.array([ATOM_TYPE_MAPPING.index(get_atom_type(atom)) for atom in atoms])
    data.x = torch.tensor(node_features, dtype=torch.long).view(-1, 1)
    
    return data

class PDBDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PDBDataset, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.pdb')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No download needed as files are already in the raw_dir
        pass

    def process(self):
        data_list = []
        raw_paths = [self.root+"/"+f for f in os.listdir(self.root) if f.endswith('.pdb')]
        for raw_path in raw_paths:
            print(raw_path)
            distance_matrix, atoms = calculate_distance_matrix(raw_path)
            graph = create_graph_from_distance_matrix(distance_matrix, atoms)
            if graph is None:
                continue
            data_list.append(graph)
        print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(idx)
            return data
        else:
            raise IndexError(f"Index {idx} is not an integer")

# Ensure the directory structure is created
root = '/home/leo/MaximData/ls_test2/raw'  # Specify the path to your PDB folder
raw_dir = os.path.join(root, 'raw')
processed_dir = os.path.join(root, 'processed')

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)



dataset = PDBDataset(root)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
#data = dataset[0]  # Use the first graph in the dataset

# Initialize the model
model = GraphVAE(in_channels=dataset.num_node_features, hidden_channels=32, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
for epoch in range(2000):
    for data in train_loader:
        model.train()
        optimizer.zero_grad()

        # Forward pass
#        print(data.edge_attr.shape)
        recon, mu, logvar = model(data)
#        print(recon)
        # Assume we use all edges as positive samples and randomly sample negative edges
        pos_edge_index = data.edge_index
        neg_edge_index = torch.randint(0, data.num_nodes, pos_edge_index.size(), dtype=torch.long)
#        pos_edge_weight = data.edge_attr
#        neg_edge_weight = torch.rand_like(pos_edge_weight)
        # Compute the loss
        recon_loss = model.recon_loss(mu, pos_edge_index , neg_edge_index) #, pos_edge_index, neg_edge_weight)
        kl_loss = model.kl_loss(mu, logvar)
        loss = recon_loss + kl_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print the loss
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')