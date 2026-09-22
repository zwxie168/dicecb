import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import set_seed_all


def parse_args():
    parser = argparse.ArgumentParser(description="Create fixed train/val/test splits.")
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Split mode: 1=random stratified, 2=grouped by cell line, 3=grouped by drug pair.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for generating multiple splits.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="splits",
        help="Output directory for the split files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./GDSC_Combo_data",
        help="Directory containing the dataset CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split directories.",
    )
    return parser.parse_args()


def _label_distribution_str(series: pd.Series) -> str:
    counts = series.value_counts().sort_index()
    ratios = (counts / counts.sum() * 100).round(2)
    parts = [f"Label {k}: {counts[k]} ({ratios[k]}%)" for k in counts.index]
    return "; ".join(parts)


def make_random_split(df: pd.DataFrame, seed: int):
    train_val, test = train_test_split(
        df,
        test_size=0.1,
        stratify=df["Label"],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=1 / 9,
        stratify=train_val["Label"],
        random_state=seed,
    )
    return train, val, test


def _majority_label(df: pd.DataFrame, group_col: str) -> pd.Series:
    return df.groupby(group_col)["Label"].apply(lambda x: x.value_counts().idxmax())


def make_grouped_split(df: pd.DataFrame, group_col: str, seed: int, stratify: bool = True):
    groups = df[group_col].unique()
    group_labels = _majority_label(df, group_col) if stratify else None

    def _stratify_for(group_list):
        return None if group_labels is None else [group_labels[g] for g in group_list]

    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=0.1,
        stratify=_stratify_for(groups),
        random_state=seed,
    )
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=1 / 9,
        stratify=_stratify_for(train_val_groups),
        random_state=seed,
    )

    train = df[df[group_col].isin(train_groups)].copy()
    val = df[df[group_col].isin(val_groups)].copy()
    test = df[df[group_col].isin(test_groups)].copy()
    return train, val, test


def save_split(split_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "train_idx.npy", train.index.values.astype(np.int64))
    np.save(split_dir / "val_idx.npy", val.index.values.astype(np.int64))
    np.save(split_dir / "test_idx.npy", test.index.values.astype(np.int64))
    train.to_csv(split_dir / "train.csv", index=False)
    val.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)


def create_splits(mode: int, seeds, out_dir: str, data_dir: str, overwrite: bool):
    synergy_file = os.path.join(data_dir, "synergy_data.csv")
    if not os.path.exists(synergy_file):
        raise FileNotFoundError(f"Synergy data file not found: {synergy_file}")

    df = pd.read_csv(synergy_file)
    df["LIBRARY_ID"] = df["LIBRARY_ID"].astype(str)
    df["ANCHOR_ID"] = df["ANCHOR_ID"].astype(str)

    if mode == 2:
        group_col = "COSMIC_ID"
        mode_name = "mode_2_cell_line"
        stratify = True
    elif mode == 3:
        df["drug_pair"] = df.apply(
            lambda r: "__".join(sorted([r["LIBRARY_ID"], r["ANCHOR_ID"]])), axis=1
        )
        group_col = "drug_pair"
        mode_name = "mode_3_drug_pair"
        stratify = False
    else:
        group_col = None
        mode_name = "mode_1_random"
        stratify = None

    out_root = Path(out_dir) / mode_name

    print(f"[Info] Creating splits with mode={mode} ({mode_name}), seeds={seeds}")
    print(f"[Info] Total samples: {len(df)}")
    print(f"[Info] Overall label distribution: {_label_distribution_str(df['Label'])}")

    meta = {
        "mode": mode,
        "mode_name": mode_name,
        "total_samples": len(df),
        "seeds": seeds,
        "splits": {},
    }

    for seed in seeds:
        set_seed_all(seed)
        split_dir = out_root / f"split_{seed}"

        if split_dir.exists() and not overwrite:
            print(f"[Warning] {split_dir} already exists, skipping. Use --overwrite to replace.")
            continue

        if mode == 1:
            train, val, test = make_random_split(df, seed)
        else:
            train, val, test = make_grouped_split(df, group_col, seed, stratify=stratify)

        save_split(split_dir, train, val, test)

        print(f"\n[Split {seed}]")
        print(f"  Train: {len(train):6d}  {_label_distribution_str(train['Label'])}")
        print(f"  Val:   {len(val):6d}  {_label_distribution_str(val['Label'])}")
        print(f"  Test:  {len(test):6d}  {_label_distribution_str(test['Label'])}")

        meta["splits"][str(seed)] = {
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "train_label_dist": train["Label"].value_counts().sort_index().to_dict(),
            "val_label_dist": val["Label"].value_counts().sort_index().to_dict(),
            "test_label_dist": test["Label"].value_counts().sort_index().to_dict(),
        }

    meta_path = out_root / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[Info] Splits saved to: {out_root}")
    print(f"[Info] Meta file saved to: {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    create_splits(
        mode=args.mode,
        seeds=args.seeds,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )
