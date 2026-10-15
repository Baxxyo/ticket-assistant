"""
Fine-tune a multilingual encoder (DistilmBERT) on English + Arabic tickets.
Needs a GPU (Colab: Runtime > Change runtime type > T4 GPU).
Run AFTER `python assistant.py eval`:
  python finetune.py
Adds row "5." to results/comparison.csv, so the demo's Results tab shows it too.
"""
import os, sys, math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding, EarlyStoppingCallback, pipeline)
from assistant import load_data, load_csv, score_row, ar_scores, bench, SEED, ARABIC_TRAIN, ARABIC_TEST

MODEL = "distilbert/distilbert-base-multilingual-cased"
OUT = "artifacts/intent-distilmbert"
NAME = "5. Fine-tuned DistilmBERT (EN + AR train)"
AR_OVERSAMPLE = 10     # Arabic is ~3% of training data; repeat it (train only, never test)
MAX_LEN = 48
BATCH, EPOCHS, LR = 32, 4, 3e-5


def learning_curve(trainer, path="results/learning_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(trainer.state.log_history)
    ev = h.dropna(subset=["eval_loss"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if "loss" in h:
        tr = h.dropna(subset=["loss"])
        ax.plot(tr.epoch, tr.loss, "o-", label="train loss")
    ax.plot(ev.epoch, ev.eval_loss, "o-", label="validation loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(ev.epoch, ev.eval_macro_f1, "s--", color="green", label="validation macro-F1")
    ax2.set_ylabel("macro-F1")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="center right", fontsize=8)
    ax.set_title("Fine-tuning learning curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    if not torch.cuda.is_available():
        print("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, re-upload the files, then rerun.")
        sys.exit(1)
    os.makedirs("results", exist_ok=True)

    # ----- data (same splits as assistant.py) -----
    train, val, test = load_data()
    ar_all, ar_test = load_csv(ARABIC_TRAIN), load_csv(ARABIC_TEST)
    if ar_all is None or ar_test is None:
        print("Upload arabic_train.csv and arabic_test.csv next to this file first.")
        sys.exit(1)
    ar_tr, ar_val = train_test_split(ar_all, test_size=0.25, stratify=ar_all.intent, random_state=SEED)
    tr_df = pd.concat([train[["text", "intent"]]] + [ar_tr[["text", "intent"]]] * AR_OVERSAMPLE,
                      ignore_index=True)
    va_df = pd.concat([val[["text", "intent"]], ar_val[["text", "intent"]]], ignore_index=True)

    labels = sorted(train.intent.unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(MODEL)

    def to_ds(df):
        ds = Dataset.from_pandas(df.assign(label=df.intent.map(label2id))[["text", "label"]],
                                 preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True)

    ds_tr, ds_va = to_ds(tr_df), to_ds(va_df)
    ds_te_en, ds_te_ar = to_ds(test[["text", "intent"]]), to_ds(ar_test[["text", "intent"]])
    print(f"train={len(ds_tr)} (Arabic x{AR_OVERSAMPLE})  val={len(ds_va)}")

    # ----- model + training -----
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id)
    total_steps = math.ceil(len(ds_tr) / BATCH) * EPOCHS

    args = TrainingArguments(
        output_dir="runs/intent", learning_rate=LR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=128,
        warmup_steps=int(0.1 * total_steps), weight_decay=0.01, seed=SEED, fp16=True,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        report_to="none")

    def compute_metrics(p):
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

    kw = dict(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
              data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
              callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    try:
        trainer = Trainer(processing_class=tok, **kw)
    except TypeError:                      # older transformers versions
        trainer = Trainer(tokenizer=tok, **kw)

    trainer.train()
    learning_curve(trainer)

    # ----- test ONCE, at the end -----
    pred_en = np.array([id2label[i] for i in trainer.predict(ds_te_en).predictions.argmax(-1)])
    pred_ar = np.array([id2label[i] for i in trainer.predict(ds_te_ar).predictions.argmax(-1)])
    row = score_row(NAME, test, pred_en, ar_test, pred_ar)

    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    clf = pipeline("text-classification", model=OUT, device=0, truncation=True, max_length=MAX_LEN)
    row.update(bench(lambda t: clf(t), test.text.tolist()[:150] + ar_test.text.tolist()))

    ar_test.assign(pred_finetuned=pred_ar).to_csv(
        "results/arabic_predictions_finetuned.csv", index=False, encoding="utf-8-sig")

    # ----- add to the comparison table -----
    path = "results/comparison.csv"
    board = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(board):
        board = board[board.model != NAME]
    board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
    board.to_csv(path, index=False)
    print("\n=== Comparison (same test sets) ===")
    print(board.to_string(index=False))

    flags = "results/ar_test_flags.csv"
    if os.path.exists(flags):
        dup = pd.read_csv(flags).near_duplicate.values.astype(bool)
        print("\nWithout near-duplicate Arabic test tickets:",
              ar_scores(ar_test[~dup], pred_ar[~dup]))
    print("\nSaved: results/learning_curve.png, results/arabic_predictions_finetuned.csv, model in", OUT)


if __name__ == "__main__":
    main()
