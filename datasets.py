# dataset.py
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import dgl
from utils import get_adjacency

class MultimodalDataset(Dataset):
    def __init__(self, root_dir):
    
        self.fc = np.load(os.path.join(root_dir, 'fc.npy'))
        self.sc = np.load(os.path.join(root_dir, 'sc.npy'))
        self.morph = np.load(os.path.join(root_dir, 'morph.npy'))
        self.cog = np.load(os.path.join(root_dir, 'cog.npy'))
        self.labels = np.load(os.path.join(root_dir, 'labels.npy'))


#        fc_dict     = np.load(os.path.join(root_dir, 'fc.npy'), allow_pickle=True).item()
#        sc_dict     = np.load(os.path.join(root_dir, 'sc.npy'), allow_pickle=True).item()
#        morph_dict  = np.load(os.path.join(root_dir, 'morph.npy'), allow_pickle=True).item()
#        cog_dict    = np.load(os.path.join(root_dir, 'cog.npy'), allow_pickle=True).item()
#        labels_dict = np.load(os.path.join(root_dir, 'labels.npy'), allow_pickle=True).item()
#
#
#        self.fc     = list(fc_dict.values())
#        self.sc     = list(sc_dict.values())
#        
#        self.morph = [
#            np.array(v.select_dtypes(include=[np.number]), dtype=np.float32)
#            if hasattr(v, "select_dtypes") else np.array(v, dtype=np.float32)
#            for v in morph_dict.values()
#        ]
#
#        self.cog = [
#            np.array(list(v.values()), dtype=np.float32)
#            if isinstance(v, dict)
#            else np.array(v.to_numpy(), dtype=np.float32) if hasattr(v, "to_numpy")
#            else np.array(v, dtype=np.float32)
#            for v in cog_dict.values()]
#        self.labels = [
#            np.array([float(x) for x in v.values()], dtype=np.float32)
#            if isinstance(v, dict)
#            else np.array([float(x) for x in v], dtype=np.float32)
#            for v in labels_dict.values()]



    def __len__(self):
        return len(self.fc)

    def __getitem__(self, idx):
        return (
            self.fc[idx],
            self.sc[idx],
            self.morph[idx],
            self.cog[idx],
            self.labels[idx]
        )


class MultimodalDGLDataset(MultimodalDataset):
    def __init__(self, root_dir, threshold=0.0, mode='topk', k=30):
        super().__init__(root_dir)
        self.threshold = threshold
        self.mode = mode
        self.k = k

    def get_adjacency(self, matrix):
        return get_adjacency(matrix, mode=self.mode, k=self.k, threshold=self.threshold)

    def build_graph(self, conn_matrix):
        adj = self.get_adjacency(conn_matrix)
        src, dst = np.nonzero(adj)
        weights = adj[src, dst]
        g = dgl.graph((src, dst), num_nodes=adj.shape[0])
        g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
        g.ndata['feat'] = torch.tensor(conn_matrix, dtype=torch.float32)
        return g

    def __getitem__(self, idx):
        fc_mat, sc_mat, morph_vec, cog_vec, label_vec = super().__getitem__(idx)

        g_fc = self.build_graph(fc_mat)
        g_sc = self.build_graph(sc_mat)

        return g_fc, g_sc, torch.tensor(morph_vec, dtype=torch.float32), \
               torch.tensor(cog_vec, dtype=torch.float32), \
               torch.tensor(label_vec, dtype=torch.float32)


def multimodal_dgl_collate_fn(batch):
    g_fc_list, g_sc_list, morphs, cogs, labels = zip(*batch)
    batched_fc = dgl.batch(g_fc_list)
    batched_sc = dgl.batch(g_sc_list)
    return (batched_fc, batched_sc,
            torch.stack(morphs), torch.stack(cogs), torch.stack(labels))

def simulate_data(N, D, d, l, c, dir = "./data/simulated_data"):
    """

    :param N: number of subjects
    :param D: number of ROIs
    :param d: morph features per ROI
    :param l: number of cognitive scores
    :param c: number of disease labels
    :param dir: directory to save data
    :return:
    """

    # Simulated data
    fc = np.random.rand(N, D, D).astype(np.float32)
    sc = np.random.rand(N, D, D).astype(np.float32)
#    morph = np.random.rand(N, D * d).astype(np.float32)  # flatten morph per subject
    morph = np.random.rand(N, D, d).astype(np.float32)  #dont flatten
    cog = np.random.rand(N, l).astype(np.float32)

    # Multi-labels: randomly assign 1–3 labels per subject
    labels = np.zeros((N, c), dtype=np.float32)
    for i in range(N):
        labels[i, np.random.choice(c, size=np.random.randint(1, 4), replace=False)] = 1

    # Save to disk
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, "fc.npy"), fc)
    np.save(os.path.join(dir, "sc.npy"), sc)
    np.save(os.path.join(dir, "morph.npy"), morph)
    np.save(os.path.join(dir, "cog.npy"), cog)
    np.save(os.path.join(dir, "labels.npy"), labels)




if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
#    simulate_data(N=100, D=400, d=10, l=10, c=6)

    full_dataset = MultimodalDGLDataset(root_dir='./data/common')
    labels = full_dataset.labels  # shape (N, C)

#    stratify_labels = labels.argmax(axis=1)  # only if one label per sample

    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
#        stratify=stratify_labels,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=multimodal_dgl_collate_fn,num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=multimodal_dgl_collate_fn,num_workers=0)
    for i in train_loader:
        print(i)
        graph_list = dgl.unbatch(i[0])
        print(graph_list[0])
        print(graph_list[0].edata['weight'].shape)
