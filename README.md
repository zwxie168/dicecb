# DIceCB: Drug Combination Response Classification

## 1. Environment setup

The code needs **Python 3.10**. `device` is selected automatically in `main.py:11` — CUDA if
available, otherwise CPU.

### Conda (Linux)

```bash
conda env create -n dicecb -f environment.yaml
conda activate dicecb
```

### Versions the reported configuration was run with

| Package      | Version |
|--------------|---------|
| Python       | 3.10.19 |
| torch        | 2.9.1   |
| transformers | 4.57.3  |
| rdkit        | 2025.03.6 |
| numpy        | 2.2.6   |
| pandas       | 2.3.3   |
| scikit-learn | 1.7.2   |

---

## 2. Data and pretrained model
Both resources are hosted on Google Drive and are **not** part of the git repository — these
directories ship empty, so the project cannot run until you download them.

### 2.1 Dataset → `./GDSC_Combo_data/`

**Download:** <https://drive.google.com/drive/folders/1BPZCY7TQyOtQ5bVjHDQvhKf_1pLFkyiT?usp=drive_link>

Place the CSVs in `./GDSC_Combo_data/`

### 2.2 ChemBERTa encoder → `./ChemBERTa-zinc-base-v1/`

**Download:** <https://drive.google.com/drive/folders/1mlvjSNoiY1nbGlPs51I3mRB1-yvXWF8G?usp=drive_link>

Place the HuggingFace-format model files directly in `./ChemBERTa-zinc-base-v1/` — no nested
subdirectory:

```
config.json
merges.txt
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
vocab.json
```

---

## 3. Running the code

Run both commands from the repository root.

### Step 1 — Generate the data splits (optional)

```bash
python create_splits.py --mode 2 --seeds 0 1 2
```

### Step 2 — Train and evaluate

```bash
python main.py
```
