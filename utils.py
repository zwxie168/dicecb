import torch
import numpy as np
import random
from sklearn.metrics import classification_report
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem


GRAPH_DISTANCE = np.array([
    [0, 1, 1, 1, 2, 3],
    [1, 0, 2, 2, 1, 2],
    [1, 2, 0, 2, 1, 2],
    [1, 2, 2, 0, 1, 2],
    [2, 1, 1, 1, 0, 1],
    [3, 2, 2, 2, 1, 0]
], dtype=np.float32)

GRAPH_DISTANCE_SQUARED = GRAPH_DISTANCE ** 2


def _qwk_graph(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    n_classes = GRAPH_DISTANCE.shape[0]

    O_counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    for yt, yp in zip(y_true, y_pred):
        O_counts[yt, yp] += 1.0
    O = O_counts / n

    row_probs = O.sum(axis=1, keepdims=True)
    col_probs = O.sum(axis=0, keepdims=True)
    E = row_probs @ col_probs

    D2 = GRAPH_DISTANCE_SQUARED.astype(np.float64)
    obs = (D2 * O).sum()
    exp = (D2 * E).sum()

    if exp < 1e-12:
        return 0.0
    return float(1.0 - obs / exp)


def metrics_ordinal(y_test, y_pred):
    y_test = np.asarray(y_test, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    d = GRAPH_DISTANCE[y_test, y_pred]
    mae_graph = float(d.mean())
    acc_at_0 = float((d <= 0).mean())
    acc_at_1 = float((d <= 1).mean())
    acc_at_2 = float((d <= 2).mean())

    qwk_graph = _qwk_graph(y_test, y_pred)

    return mae_graph, qwk_graph, acc_at_0, acc_at_1, acc_at_2


def get_Morgan(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_report_cls_tissue(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            emb1 = batch["emb1"].to(device)
            emb2 = batch["emb2"].to(device)
            fp1 = batch['fp1'].to(device)
            fp2 = batch['fp2'].to(device)
            gene = batch["gene"].to(device)
            tissue_id = batch["tissue_id"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs, _ = model(emb1, emb2, gene, fp1, fp2, tissue_id)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(labels.tolist())

    mae_graph, qwk_graph, acc_at_0, acc_at_1, acc_at_2 = metrics_ordinal(all_trues, all_preds)
    target_names = ['sub-HSA', 'SA-A', 'SA-L', 'IDA', 'Bliss', 'Synergy']
    report = classification_report(all_trues, all_preds, target_names=target_names, zero_division=0)
    return mae_graph, qwk_graph, acc_at_0, acc_at_1, acc_at_2, report
