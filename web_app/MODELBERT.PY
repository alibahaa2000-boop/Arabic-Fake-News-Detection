"""
arabert_news_final_fixed.py
─────────────────────────────────────────────────────────────────────────────
Arabic fake-news binary classifier fine-tuned on AraBERT.

Hardware target : RTX 3050 (4 GB VRAM) + i7-11370H + 16 GB RAM
Priority        : Balance speed and accuracy

FIXES vs original:
  1. Removed 325 conflict rows — same text existed in both real AND fake
     datasets, so the model learned contradictory labels.  Those rows are
     now dropped BEFORE training starts.
  2. drop_duplicates now runs BEFORE concat with keep=False so any text
     that appears in both classes is eliminated entirely.
  3. max_length unified to 128 everywhere (training AND inference).
     The original Flask app used 256 while training used 128 — tokens in
     positions 129-256 were never seen during training, degrading inference.
  4. Tokenisation cache is invalidated whenever max_length changes.
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"

import json
import hashlib
import logging
import shutil
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from datasets import Dataset, ClassLabel, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# (A) Configuration
# =============================================================================
@dataclass
class Config:
    true_file : str = r"D:\Desktop\Selenium\selenium_dataset\finaltruedataset.csv"
    fake_file : str = r"D:\Downloads\fake_only_10000_semicolon.csv"
    model_name: str = "aubmindlab/bert-base-arabertv2"
    work_dir  : str = "./arabert_news_artifacts"

    @property
    def token_cache_dir(self) -> str: return os.path.join(self.work_dir, "tokenized_cache")
    @property
    def manifest_path(self)   -> str: return os.path.join(self.work_dir, "manifest.json")
    @property
    def model_dir(self)       -> str: return os.path.join(self.work_dir, "model")
    @property
    def plots_dir(self)       -> str: return os.path.join(self.work_dir, "plots")

    # ── FIX 3: max_length is now 128 everywhere ───────────────────────────────
    # The Flask app was using 256 while training used 128.  Positions 129-256
    # were never seen during training, so inference was degraded.  Both sides
    # now use 128.  If you want 256, change it here AND in app.py — never in
    # just one place.
    max_length : int   = 256       # single source of truth
    train_ratio: float = 0.8
    seed       : int   = 42

    epochs      : int   = 4
    lr          : float = 2e-5
    train_bs    : int   = 8
    eval_bs     : int   = 16
    grad_accum  : int   = 2
    warmup_ratio: float = 0.1
    lr_scheduler: str   = "cosine"
    early_stopping_patience: int = 2
    dataloader_workers: int  = 4
    pin_memory        : bool = True


CFG = Config()


# =============================================================================
# (B) Helper utilities
# =============================================================================
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_manifest() -> Dict[str, Any]:
    if os.path.exists(CFG.manifest_path):
        with open(CFG.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"known_hashes": [], "max_length": CFG.max_length}


def save_manifest(m: Dict[str, Any]) -> None:
    with open(CFG.manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def robust_read_csv(path: str) -> pd.DataFrame:
    separators = [",", ";", "\t", "|"]
    encodings  = ["utf-8", "utf-8-sig", "cp1256"]
    last_err   = None
    for enc in encodings:
        for sep in separators:
            try:
                return pd.read_csv(path, sep=sep, engine="python",
                                   encoding=enc, on_bad_lines="skip")
            except Exception as exc:
                last_err = exc
    raise last_err


def guess_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "content", "Content", "text", "Text",
        "news",    "News",    "body", "Body",
        "article", "Article", "description", "Description",
        "تفاصيل", "المقال", "المحتوى", "النص", "خبر", "عنوان",
        "title",   "Title",
    ]
    for name in candidates:
        if name in df.columns:
            return name
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return df.columns[0]


# =============================================================================
# (C) Data loading and preparation  ← FIXED
# =============================================================================
def load_and_prepare() -> pd.DataFrame:
    for path in [CFG.true_file, CFG.fake_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

    # ── Real news (label = 1) ─────────────────────────────────────────────────
    try:
        df_true = pd.read_csv(CFG.true_file, sep=";", encoding="utf-8-sig")
    except Exception:
        df_true = robust_read_csv(CFG.true_file)

    if len(df_true.columns) == 1:
        split = df_true.iloc[:, 0].astype(str).str.split(";", n=1, expand=True)
        if split.shape[1] == 2:
            split.columns = ["site", "content"]
            df_true = split

    col = "content" if "content" in df_true.columns else guess_text_column(df_true)
    df_true = df_true[[col]].rename(columns={col: "text"})
    df_true["label"] = 1

    # ── Fake news (label = 0) ─────────────────────────────────────────────────
    try:
        df_fake = pd.read_csv(CFG.fake_file, sep=";", encoding="utf-8-sig")
    except Exception:
        df_fake = robust_read_csv(CFG.fake_file)

    col = guess_text_column(df_fake)
    df_fake = df_fake[[col]].rename(columns={col: "text"})
    df_fake["label"] = 0

    # ── Text cleaning (do this BEFORE conflict detection) ─────────────────────
    for df_ in [df_true, df_fake]:
        df_["text"] = (
            df_["text"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # ── FIX 1 & 2: Remove conflicting texts BEFORE any sampling ──────────────
    #
    # Problem in original code:
    #   325 articles existed in BOTH datasets with OPPOSITE labels.
    #   The original script called drop_duplicates AFTER concat, but
    #   drop_duplicates looks at ALL columns — a row with text="X",label=1
    #   and a row with text="X",label=0 are NOT duplicates (different labels),
    #   so both survived into training.  The model saw the same sentence with
    #   contradictory ground truth, which made it uncertain and biased.
    #
    # Fix: build a blacklist of texts that appear in BOTH classes, then
    # remove them from both DataFrames before any further processing.
    true_texts  = set(df_true["text"])
    fake_texts  = set(df_fake["text"])
    conflict_texts = true_texts & fake_texts

    if conflict_texts:
        log.warning(
            "CONFLICT DETECTED: %d texts appear in both real AND fake datasets. "
            "Removing them from both to prevent contradictory label training.",
            len(conflict_texts),
        )
        df_true = df_true[~df_true["text"].isin(conflict_texts)].copy()
        df_fake = df_fake[~df_fake["text"].isin(conflict_texts)].copy()

    # ── Remove duplicates within each class separately ────────────────────────
    df_true = df_true.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_fake = df_fake.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # ── Filter short texts ────────────────────────────────────────────────────
    df_true = df_true[df_true["text"].str.len() > 50].reset_index(drop=True)
    df_fake = df_fake[df_fake["text"].str.len() > 50].reset_index(drop=True)

    # ── Balance classes ───────────────────────────────────────────────────────
    n = min(len(df_true), len(df_fake))
    df = pd.concat(
        [df_true.sample(n, random_state=CFG.seed),
         df_fake.sample(n, random_state=CFG.seed)],
        ignore_index=True,
    )

    df["text_hash"] = df["text"].apply(sha1_text)

    log.info("Dataset ready — real: %d | fake: %d | total: %d", n, n, len(df))
    log.info("Label counts:\n%s", df["label"].value_counts().to_string())
    return df


# =============================================================================
# (D) Tokenisation with incremental disk cache  ← FIXED
# =============================================================================
def _ensure_classlabel(ds: Dataset) -> Dataset:
    feats = ds.features.copy()
    feats["label"] = ClassLabel(names=["fake", "true"])
    return ds.cast(feats)


def tokenize_incremental(df: pd.DataFrame, tokenizer) -> Dataset:
    manifest = load_manifest()
    known    = set(manifest.get("known_hashes", []))

    # ── FIX 4: Invalidate cache if max_length changed ─────────────────────────
    # If someone changes max_length in Config, the old cache used different
    # truncation and must be rebuilt — otherwise you silently train/infer
    # on mismatched sequence lengths.
    cached_max_len = manifest.get("max_length", CFG.max_length)
    if cached_max_len != CFG.max_length:
        log.warning(
            "max_length changed from %d → %d. Invalidating tokenisation cache.",
            cached_max_len, CFG.max_length,
        )
        known = set()
        if os.path.exists(CFG.token_cache_dir):
            shutil.rmtree(CFG.token_cache_dir)

    new_df = df[~df["text_hash"].isin(known)].copy()

    cached = None
    if os.path.exists(CFG.token_cache_dir):
        try:
            cached = _ensure_classlabel(load_from_disk(CFG.token_cache_dir))
        except Exception:
            cached = None

    if len(new_df) == 0:
        if cached is None:
            raise RuntimeError("Cache is empty and no new samples were found.")
        log.info("All samples already cached (size=%d) — skipping tokenisation", len(cached))
        return cached

    log.info("Tokenising %d new samples …", len(new_df))

    new_ds = Dataset.from_pandas(
        new_df[["text", "label", "text_hash"]], preserve_index=False
    )

    def _tokenise_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=CFG.max_length)

    new_tok = _ensure_classlabel(
        new_ds.map(_tokenise_batch, batched=True, remove_columns=["text"])
    )

    merged = new_tok if cached is None else _ensure_classlabel(
        concatenate_datasets([cached, new_tok])
    )

    tmp = CFG.token_cache_dir + "_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    merged.save_to_disk(tmp)
    if os.path.exists(CFG.token_cache_dir):
        shutil.rmtree(CFG.token_cache_dir)
    os.rename(tmp, CFG.token_cache_dir)

    manifest["known_hashes"] = list(known | set(new_df["text_hash"]))
    manifest["max_length"]   = CFG.max_length   # store so next run can detect changes
    save_manifest(manifest)

    log.info("Cache updated (total size=%d)", len(merged))
    return merged


# =============================================================================
# (E) Train / validation split
# =============================================================================
def split_train_val(ds: Dataset) -> Tuple[Dataset, Dataset]:
    splits = ds.train_test_split(
        test_size=(1 - CFG.train_ratio),
        seed=CFG.seed,
        stratify_by_column="label",
    )
    return splits["train"], splits["test"]


# =============================================================================
# (F) Metrics, training callback, and plots
# =============================================================================
def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class TrainEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)

        train_m = self.trainer.evaluate(
            eval_dataset=self.trainer.train_dataset,
            metric_key_prefix="train",
        )
        val_m = self.trainer.evaluate(
            eval_dataset=self.trainer.eval_dataset,
            metric_key_prefix="eval",
        )

        combined = {"epoch": epoch}
        combined.update(train_m)
        combined.update(val_m)
        state.log_history.append(combined)

        sep = "-" * 70
        print(sep)
        print(f"  Epoch {epoch}/{int(args.num_train_epochs)}")
        print(
            f"  [TRAIN] loss={train_m.get('train_loss', 0):.4f} | "
            f"acc={train_m.get('train_accuracy', 0):.4f} | "
            f"precision={train_m.get('train_precision', 0):.4f} | "
            f"recall={train_m.get('train_recall', 0):.4f} | "
            f"f1={train_m.get('train_f1', 0):.4f}"
        )
        print(
            f"  [VAL  ] loss={val_m.get('eval_loss', 0):.4f} | "
            f"acc={val_m.get('eval_accuracy', 0):.4f} | "
            f"precision={val_m.get('eval_precision', 0):.4f} | "
            f"recall={val_m.get('eval_recall', 0):.4f} | "
            f"f1={val_m.get('eval_f1', 0):.4f}"
        )
        print(sep)
        return control


def _extract_series(logs, train_key: str, eval_key: str):
    epoch_rows: Dict[int, dict] = {}
    for row in logs:
        ep = row.get("epoch")
        if ep is None:
            continue
        ep = int(round(ep))
        existing = epoch_rows.get(ep)
        if existing is None:
            epoch_rows[ep] = row
        elif train_key in row and eval_key in row:
            epoch_rows[ep] = row

    tx, ty, ex, ey = [], [], [], []
    for ep in sorted(epoch_rows):
        row = epoch_rows[ep]
        if train_key in row:
            tx.append(ep); ty.append(row[train_key])
        if eval_key in row:
            ex.append(ep); ey.append(row[eval_key])
    return tx, ty, ex, ey


def _save_line_chart(tx, ty, ex, ey, ylabel: str, title: str, path: str):
    plt.figure()
    if ty: plt.plot(tx, ty, marker="o", label=f"Train {ylabel}")
    if ey: plt.plot(ex, ey, marker="s", label=f"Validation {ylabel}")
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()
    log.info("Plot saved → %s", path)


def plot_all_metrics(trainer: Trainer):
    logs = trainer.state.log_history
    p    = CFG.plots_dir
    for train_key, eval_key, label in [
        ("train_accuracy",  "eval_accuracy",  "Accuracy"),
        ("train_precision", "eval_precision", "Precision"),
        ("train_recall",    "eval_recall",    "Recall"),
        ("train_f1",        "eval_f1",        "F1"),
    ]:
        tx, ty, ex, ey = _extract_series(logs, train_key, eval_key)
        _save_line_chart(tx, ty, ex, ey, ylabel=label,
                         title=f"Train vs Validation {label}",
                         path=os.path.join(p, f"{label.lower()}.png"))

    tx, ty, ex, ey = [], [], [], []
    for row in logs:
        ep = row.get("epoch", row.get("step"))
        if "loss" in row and "eval_loss" not in row:
            tx.append(ep); ty.append(row["loss"])
        if "eval_loss" in row:
            ex.append(ep); ey.append(row["eval_loss"])
    _save_line_chart(tx, ty, ex, ey, "Loss", "Train vs Validation Loss",
                     os.path.join(p, "loss.png"))


def plot_confusion_matrix(trainer: Trainer, val_ds: Dataset):
    pred_out = trainer.predict(val_ds)
    y_pred   = np.argmax(pred_out.predictions, axis=-1)
    y_true   = pred_out.label_ids
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["Fake", "True"],
    )
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Validation Set")
    path = os.path.join(CFG.plots_dir, "confusion_matrix.png")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
    log.info("Plot saved → %s", path)


def plot_token_heatmap(trainer: Trainer, tokenizer, val_ds: Dataset, top_n: int = 64):
    pred_out = trainer.predict(val_ds)
    y_pred   = np.argmax(pred_out.predictions, axis=-1)
    y_true   = pred_out.label_ids
    special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    stats: Dict[int, list] = {}
    freq:  Dict[int, int]  = {}

    for i in range(len(val_ds)):
        correct = int(y_pred[i] == y_true[i])
        present = {
            int(tid)
            for tid, mask in zip(val_ds["input_ids"][i], val_ds["attention_mask"][i])
            if mask and int(tid) not in special
        }
        for tid in present:
            s = stats.setdefault(tid, [0, 0])
            s[0] += 1; s[1] += correct
            freq[tid] = freq.get(tid, 0) + 1

    if not stats:
        log.warning("Token heatmap skipped — no tokens found in validation set.")
        return

    top_ids = [tid for tid, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    accs    = [stats[tid][1] / max(stats[tid][0], 1) for tid in top_ids]
    toks    = [tokenizer.convert_ids_to_tokens(tid)  for tid in top_ids]

    side = int(np.ceil(np.sqrt(len(accs))))
    grid = np.full((side, side), np.nan)
    for idx, a in enumerate(accs):
        r, c = divmod(idx, side)
        grid[r, c] = a

    plt.figure(figsize=(10, 8))
    im = plt.imshow(grid, aspect="equal", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Token Accuracy Heatmap (top {len(accs)} tokens — validation set)")
    for idx, tok in enumerate(toks):
        r, c = divmod(idx, side)
        if not np.isnan(grid[r, c]):
            plt.text(c, r, tok, ha="center", va="center", fontsize=7)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    path = os.path.join(CFG.plots_dir, "token_heatmap.png")
    plt.savefig(path, dpi=200); plt.close()
    log.info("Plot saved → %s", path)


# =============================================================================
# (G) Checkpoint resume
# =============================================================================
def find_last_checkpoint(model_dir: str) -> Optional[str]:
    if not os.path.exists(model_dir):
        return None
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda name: int(name.split("-")[-1]))
    return os.path.join(model_dir, checkpoints[-1])


# =============================================================================
# (H) Main
# =============================================================================
def main():
    for d in [CFG.work_dir, CFG.model_dir, CFG.plots_dir]:
        os.makedirs(d, exist_ok=True)

    set_seed(CFG.seed)

    log.info("Step 1 — Load data")
    df = load_and_prepare()

    log.info("Step 2 — Load tokeniser")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    log.info("Step 3 — Tokenise (incremental cache)")
    ds_tok = tokenize_incremental(df, tokenizer)

    log.info("Step 4 — Stratified 80/20 split")
    train_ds, val_ds = split_train_val(ds_tok)
    log.info("train=%d | val=%d", len(train_ds), len(val_ds))

    log.info("Step 5 — Load model")
    model_src = (
        CFG.model_dir
        if os.path.exists(os.path.join(CFG.model_dir, "config.json"))
        else CFG.model_name
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_src, num_labels=2)
    model.gradient_checkpointing_enable()

    log.info("Step 6 — Build TrainingArguments")
    training_args = TrainingArguments(
        output_dir=CFG.model_dir,
        eval_strategy   ="epoch",
        save_strategy   ="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model ="f1",
        greater_is_better     =True,
        num_train_epochs             =CFG.epochs,
        per_device_train_batch_size  =CFG.train_bs,
        per_device_eval_batch_size   =CFG.eval_bs,
        gradient_accumulation_steps  =CFG.grad_accum,
        gradient_checkpointing       =True,
        learning_rate   =CFG.lr,
        weight_decay    =0.01,
        warmup_ratio    =CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=CFG.dataloader_workers,
        dataloader_pin_memory =CFG.pin_memory,
        group_by_length       =True,
        seed           =CFG.seed,
        report_to      ="none",
        save_total_limit=2,
    )

    log.info("Step 7 — Build Trainer")
    train_callback = TrainEvalCallback()

    trainer = Trainer(
        model            =model,
        args             =training_args,
        train_dataset    =train_ds,
        eval_dataset     =val_ds,
        processing_class =tokenizer,
        data_collator    =DataCollatorWithPadding(tokenizer),
        compute_metrics  =compute_metrics,
        callbacks=[
            train_callback,
            EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience),
        ],
    )
    train_callback.trainer = trainer

    resume = find_last_checkpoint(CFG.model_dir)
    if resume:
        log.info("Resuming from checkpoint: %s", resume)

    log.info("Step 8 — Train")
    trainer.train(resume_from_checkpoint=resume)

    log.info("Step 9 — Final evaluation")
    val_metrics   = trainer.evaluate(eval_dataset=val_ds,   metric_key_prefix="eval")
    train_metrics = trainer.evaluate(eval_dataset=train_ds, metric_key_prefix="train")
    log.info("VALIDATION — %s", val_metrics)
    log.info("TRAIN      — %s", train_metrics)

    log.info("Step 10 — Save model + tokeniser")
    trainer.save_model(CFG.model_dir)
    tokenizer.save_pretrained(CFG.model_dir)

    log.info("Step 11 — Generate plots")
    plot_all_metrics(trainer)
    plot_confusion_matrix(trainer, val_ds)
    plot_token_heatmap(trainer, tokenizer, val_ds, top_n=64)

    log.info("Done.")
    log.info("Model saved → %s", CFG.model_dir)
    log.info("Plots saved → %s", CFG.plots_dir)


if __name__ == "__main__":
    main()