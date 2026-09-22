import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from utils import get_Morgan


@torch.no_grad()
def drug_feature(unique_smiles, model_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smiles_embedding_dict = {}

    if model_dir is not None and os.path.isdir(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        bert_model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    else:
        SMILES_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(SMILES_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(SMILES_MODEL_NAME)

    bert_model.eval().to(device)

    for sm in unique_smiles:
        encoded = tokenizer(sm, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        smiles_embedding_dict[sm] = cls_embedding

    return smiles_embedding_dict


class CLSSynergyDataset(Dataset):

    def __init__(self, df, gene_df, smiles_embedding_dict,
                 tissue_to_id=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.gene_df = gene_df.set_index('COSMIC_ID')
        self.smiles_embedding_dict = smiles_embedding_dict
        self.transform = transform
        if tissue_to_id is None:
            tissues = sorted(self.df['Tissue'].unique())
            tissue_to_id = {t: i for i, t in enumerate(tissues)}
        self.tissue_to_id = tissue_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        smiles1 = row['LIBRARY_ID']
        smiles2 = row['ANCHOR_ID']
        cosmic_id = row['COSMIC_ID']

        emb1 = torch.tensor(
            self.smiles_embedding_dict[smiles1], dtype=torch.float32)
        emb2 = torch.tensor(
            self.smiles_embedding_dict[smiles2], dtype=torch.float32)

        fp1 = torch.tensor(get_Morgan(smiles1), dtype=torch.float32)
        fp2 = torch.tensor(get_Morgan(smiles2), dtype=torch.float32)

        gene_vec = torch.tensor(
            self.gene_df.loc[cosmic_id].values.astype(np.float32),
            dtype=torch.float32
        )
        tissue_id = torch.tensor(
            self.tissue_to_id[row['Tissue']], dtype=torch.long
        )
        label = torch.tensor(row['Label'], dtype=torch.long)

        sample = {
            'emb1': emb1,
            'emb2': emb2,
            'fp1': fp1,
            'fp2': fp2,
            'gene': gene_vec,
            'tissue_id': tissue_id,
            'label': label
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
