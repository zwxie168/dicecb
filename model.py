import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDistanceLoss(nn.Module):

    def __init__(self, num_classes: int = 6,
                 con_temperature: float = 0.8,
                 graph_temperature: float = 1.0,
                 w_con: float = 0.1,
                 w_graph: float = 0.1,
                 lam_geo: float = 0.01,
                 class_weight: torch.Tensor = None):
        super().__init__()

        if w_con + w_graph + lam_geo >= 1.0:
            raise ValueError(
                f"w_con ({w_con}) + w_graph ({w_graph}) + lam_geo ({lam_geo}) "
                f"must be < 1.0 so that the cross-entropy term has positive weight."
            )

        self.num_classes = num_classes
        self.con_temperature = con_temperature
        self.graph_temperature = graph_temperature
        self.w_con = w_con
        self.w_graph = w_graph
        self.lam_geo = lam_geo
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss()

        penalty_matrix = torch.tensor([
            [0, 1, 1, 1, 4, 9],
            [1, 0, 4, 4, 1, 4],
            [1, 4, 0, 4, 1, 4],
            [1, 4, 4, 0, 1, 4],
            [4, 1, 1, 1, 0, 1],
            [9, 4, 4, 4, 1, 0]
        ], dtype=torch.float32)

        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(self, logits: torch.Tensor, features: torch.Tensor,
                labels: torch.LongTensor,
                cls_weight: torch.Tensor = None) -> torch.Tensor:
        device = logits.device
        B, C = logits.shape

        L_ce = self.cross_entropy(logits, labels)

        z = F.normalize(features, p=2, dim=1)
        sim = (z @ z.T) / self.con_temperature
        mask_self = torch.eye(B, device=device).bool()
        sim = sim.masked_fill(mask_self, float('-inf'))

        lbl = labels.view(-1, 1)
        mask_pos = (lbl == lbl.T).float().to(device)
        mask_pos = mask_pos.masked_fill(mask_self, 0.0)

        exp_sim = sim.exp()
        denom = exp_sim.sum(dim=1)

        if self.class_weight is not None:
            w = self.class_weight.to(device)
            wj = w[labels].view(1, B)
            num = (exp_sim * mask_pos * wj).sum(dim=1)
        else:
            num = (exp_sim * mask_pos).sum(dim=1)

        eps = 1e-8
        frac = num / (denom + eps)
        L_con_each = -torch.log(frac + eps)
        L_con = L_con_each.mean()

        probs = F.softmax(logits / self.graph_temperature, dim=1)
        penalties = self.penalty_matrix.to(device)[labels, :]
        L_graph = (probs * penalties).sum(dim=1).mean()

        if cls_weight is not None:
            wn = F.normalize(cls_weight, p=2, dim=1, eps=1e-8)
            S = wn @ wn.T

            target = 1.0 - self.penalty_matrix.to(device) / self.penalty_matrix.max()

            mask_off = ~torch.eye(self.num_classes, dtype=torch.bool, device=device)
            L_geo = ((S - target)[mask_off] ** 2).mean()
        else:
            L_geo = 0.0

        loss = ( L_ce + self.w_con * L_con + self.w_graph * L_graph )
        return loss


class CLSTissueContextSynergyModel(nn.Module):

    def __init__(self,
                 emb_dim=768,
                 gene_dim=600,
                 fp_dim=512,
                 hidden_dim=256,
                 num_classes=6,
                 num_tissues=21):
        super().__init__()

        self.drug1_fc=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.drug2_fc=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.fp1_fc=nn.Sequential(
            nn.Linear(fp_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.fp2_fc=nn.Sequential(
            nn.Linear(fp_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.gene_fc=nn.Sequential(
            nn.Linear(gene_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.tissue_emb=nn.Embedding(
            num_tissues,
            hidden_dim
        )

        self.pos_emb=nn.Parameter(
            torch.zeros(1,3,hidden_dim)
        )

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True,
            activation="relu"
        )

        self.encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.attn_pool=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2,1)
        )

        self.interaction_proj=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.gamma=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )

        self.beta=nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.classifier=nn.Sequential(
            nn.Linear(hidden_dim*4,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim,hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(hidden_dim//4),

            nn.Linear(hidden_dim//4,num_classes)
        )

    def forward(self,
                emb1,
                emb2,
                gene,
                fp1,
                fp2,
                tissue_id):

        emb1=self.drug1_fc(emb1)
        emb2=self.drug2_fc(emb2)

        fp1=self.fp1_fc(fp1)
        fp2=self.fp2_fc(fp2)

        gene=self.gene_fc(gene)

        drug1=emb1+fp1
        drug2=emb2+fp2

        tokens=torch.stack(
            [gene,drug1,drug2],
            dim=1
        )

        tokens=tokens+self.pos_emb

        hidden=self.encoder(tokens)

        score=self.attn_pool(hidden)

        weight=torch.softmax(score,dim=1)

        interaction=(weight*hidden).sum(dim=1)

        interaction=self.interaction_proj(
            interaction
        )

        tissue=self.tissue_emb(
            tissue_id
        )

        gamma=self.gamma(tissue)
        beta=self.beta(tissue)

        interaction=(1+gamma)*interaction+beta

        x=torch.cat(
            [
                gene,
                drug1,
                drug2,
                interaction
            ],
            dim=1
        )

        out=self.classifier(x)

        return out,x
