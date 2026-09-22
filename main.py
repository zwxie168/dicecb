import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from process_data import drug_feature, CLSSynergyDataset
from utils import make_report_cls_tissue, set_seed_all
from model import CLSTissueContextSynergyModel, GraphDistanceLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ep_number = 50
batch_size = 256
lr = 0.001
weight_decay = 1e-5
set_seed_all(42)

drug_file = './GDSC_Combo_data/drug_info.csv'
cline_file = './GDSC_Combo_data/omics_expression.csv'
synergy_file = './GDSC_Combo_data/synergy_data.csv'

split_dir = Path('splits/mode_2_cell_line')
split_seeds = [0, 1, 2]
results_dir = Path('Results/dicecb_mode2')
results_dir.mkdir(parents=True, exist_ok=True)

synergy_df = pd.read_csv(synergy_file)
drug_df = pd.read_csv(drug_file)
gene_df = pd.read_csv(cline_file)
synergy_df['LIBRARY_ID'] = synergy_df['LIBRARY_ID'].astype(str)
synergy_df['ANCHOR_ID'] = synergy_df['ANCHOR_ID'].astype(str)

id_to_value = dict(zip(drug_df['Pubchem_ID'].astype(str), drug_df['SMILES']))
for idx, row in synergy_df.iterrows():
    lib_drug = row['LIBRARY_ID']
    anc_drug = row['ANCHOR_ID']
    synergy_df.at[idx, 'LIBRARY_ID'] = id_to_value.get(lib_drug, lib_drug)
    synergy_df.at[idx, 'ANCHOR_ID'] = id_to_value.get(anc_drug, anc_drug)

tissues = sorted(synergy_df['Tissue'].unique())
tissue_to_id = {t: i for i, t in enumerate(tissues)}
num_tissues = len(tissues)
print(f"[Info] total {num_tissues} tissue categories: {tissues}")

unique_smiles = pd.concat([synergy_df["LIBRARY_ID"], synergy_df["ANCHOR_ID"]]).unique()
print(f"[Info] total {len(unique_smiles)} drug SMILES，initialize embedding ...")
smiles_embedding_dict = drug_feature(unique_smiles, model_dir='./ChemBERTa-zinc-base-v1')

split_results = []

for seed in split_seeds:
    split_path = split_dir / f'split_{seed}'
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_path}. "
            f"Please run 'python create_splits.py' first."
        )

    print(f"\n===== Split {seed} =====")
    train_idx = np.load(split_path / 'train_idx.npy')
    val_idx = np.load(split_path / 'val_idx.npy')
    test_idx = np.load(split_path / 'test_idx.npy')

    train_df = synergy_df.iloc[train_idx].reset_index(drop=True)
    val_df = synergy_df.iloc[val_idx].reset_index(drop=True)
    test_df = synergy_df.iloc[test_idx].reset_index(drop=True)

    print(f"[Info] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = CLSSynergyDataset(train_df, gene_df, smiles_embedding_dict, tissue_to_id)
    val_ds = CLSSynergyDataset(val_df, gene_df, smiles_embedding_dict, tissue_to_id)
    test_ds = CLSSynergyDataset(test_df, gene_df, smiles_embedding_dict, tissue_to_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    vc = train_df['Label'].value_counts().sort_index()
    cw = torch.tensor((vc.sum() / (vc + 1e-6)).values, dtype=torch.float32).to(device)

    model = CLSTissueContextSynergyModel(
        emb_dim=768,
        gene_dim=gene_df.shape[1] - 1,
        fp_dim=512,
        hidden_dim=256,
        num_classes=len(synergy_df['Label'].unique()),
        num_tissues=num_tissues
    ).to(device)

    criterion = GraphDistanceLoss(
        num_classes=6,
        con_temperature=0.8,
        graph_temperature=1.0,
        w_con=0.1,
        w_graph=0.1,
        class_weight=cw
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=ep_number)

    best_val_mae = np.inf
    best_model_path = results_dir / f'best_model_split{seed}.pt'

    for epoch in range(0, ep_number):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            emb1 = batch["emb1"].to(device)
            emb2 = batch["emb2"].to(device)
            fp1 = batch['fp1'].to(device)
            fp2 = batch['fp2'].to(device)
            gene = batch["gene"].to(device)
            tissue_id = batch["tissue_id"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs, embed = model(emb1, emb2, gene, fp1, fp2, tissue_id)
            loss = criterion(outputs, embed, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        val_mae_graph, val_qwk_graph, val_acc_at_0, val_acc_at_1, val_acc_at_2, _ = make_report_cls_tissue(model, val_loader, device)
        print('Epoch:{:3d},'.format(epoch), 'loss_train: {:.4f},'.format(avg_loss),
              'val_MAE: {:.4f},'.format(val_mae_graph), 'val_QWK: {:.4f},'.format(val_qwk_graph),
              'val_ACC@1: {:.4f}'.format(val_acc_at_1))

        if val_mae_graph < best_val_mae:
            best_val_mae = val_mae_graph
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    print('\n[Info] Best validation MAE_graph: {:.4f}'.format(best_val_mae))
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    mae_graph, qwk_graph, acc_at_0, acc_at_1, acc_at_2, report = make_report_cls_tissue(model, test_loader, device)
    print(f"\n--- Split {seed} Test Report ---\n{report}")
    print('MAE: {:.4f},'.format(mae_graph), 'QWK: {:.4f},'.format(qwk_graph), 'ACC@1: {:.4f}'.format(acc_at_1))
    split_results.append({"seed": seed, "mae": mae_graph,
                          "qwk": qwk_graph,
                          "acc_at_0": acc_at_0, "acc_at_1": acc_at_1,
                          "acc_at_2": acc_at_2})

res_df = pd.DataFrame(split_results)
res_df.to_csv(results_dir / 'model_performance.csv', index=False)
print("\n===== Fixed Split Summary =====")
print(res_df)
print(f"\n[Info] Results saved to: {results_dir / 'model_performance.csv'}")
