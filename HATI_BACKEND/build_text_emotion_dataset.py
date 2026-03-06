# build_text_emotion_dataset.py
# pip install datasets pandas scikit-learn

import os
from datasets import load_dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_LABELS = ["anger", "happy", "sad", "neutral", "anxious", "disgust", "surprised"]
OLD_SOURCES = {"go_emotions", "emotion"}
NEW_SOURCES = {"meld", "semeval2018"}
OLD_DATA_KEEP_FRAC = 0.20
HF_CACHE_DIR = os.path.join(os.getcwd(), "hf_cache")

# Common normalization
def norm_label(lbl: str):
    if lbl is None:
        return None
    l = str(lbl).strip().lower()

    if l in {"anger", "angry", "annoyance"}:
        return "anger"
    if l in {"joy", "happy", "happiness", "love"}:
        return "happy"
    if l in {"sadness", "sad", "disappointment", "grief"}:
        return "sad"
    if l in {"neutral"}:
        return "neutral"
    if l in {"fear", "nervousness", "anxiety", "anxious"}:
        return "anxious"  # <- your requested mapping
    if l in {"surprise", "surprised", "excitement"}:
        return "surprised"

    if l == "disgust":
        return "disgust"

    return None


def safe_load(candidates):
    """
    candidates = [
      ("dataset_id", None or "config"),
      ...
    ]
    """
    for ds_id, cfg in candidates:
        try:
            os.makedirs(HF_CACHE_DIR, exist_ok=True)
            ds = (
                load_dataset(ds_id, cfg, cache_dir=HF_CACHE_DIR)
                if cfg
                else load_dataset(ds_id, cache_dir=HF_CACHE_DIR)
            )
            print(f"[OK] {ds_id} config={cfg}")
            return ds_id, cfg, ds
        except Exception as e:
            print(f"[SKIP] {ds_id} config={cfg}: {e}")
    return None, None, None


def collect_rows_goemotions(ds):
    rows = []
    if "train" not in ds:
        return rows
    # GoEmotions raw has one-hot columns in your current pipeline
    keys = ds["train"].column_names
    for x in ds["train"]:
        txt = x.get("text")
        if not txt:
            continue
        mapped = None
        if "fear" in keys and x.get("fear") == 1:
            mapped = "anxious"
        elif "nervousness" in keys and x.get("nervousness") == 1:
            mapped = "anxious"
        elif "joy" in keys and x.get("joy") == 1:
            mapped = "happy"
        elif "love" in keys and x.get("love") == 1:
            mapped = "happy"
        elif "anger" in keys and x.get("anger") == 1:
            mapped = "anger"
        elif "annoyance" in keys and x.get("annoyance") == 1:
            mapped = "anger"
        elif "disgust" in keys and x.get("disgust") == 1:
            mapped = "disgust"
        elif "disapproval" in keys and x.get("disapproval") == 1:
            mapped = "disgust"
        elif "sadness" in keys and x.get("sadness") == 1:
            mapped = "sad"
        elif "neutral" in keys and x.get("neutral") == 1:
            mapped = "neutral"
        elif "surprise" in keys and x.get("surprise") == 1:
            mapped = "surprised"
        elif "excitement" in keys and x.get("excitement") == 1:
            mapped = "surprised"

        if mapped in TARGET_LABELS:
            rows.append({"text": txt, "label": mapped, "source": "go_emotions"})
    return rows


def collect_rows_generic_classlabel(ds, source_name):
    rows = []
    split = "train" if "train" in ds else list(ds.keys())[0]
    names = ds[split].features.get("label").names if "label" in ds[split].features else None
    for x in ds[split]:
        txt = x.get("text") or x.get("sentence") or x.get("tweet") or x.get("utterance")
        if not txt:
            continue
        raw = None
        if "emotion" in x:
            raw = x["emotion"]
        elif "label" in x:
            raw = names[x["label"]] if names else x["label"]
        lbl = norm_label(raw)
        if lbl in TARGET_LABELS:
            rows.append({"text": txt, "label": lbl, "source": source_name})
    return rows


def balance_per_class(df):
    # Strict balance: every class gets the same count (minority class size).
    # This avoids hidden imbalance after preprocessing/merging sources.
    class_counts = df["label"].value_counts()
    min_count = class_counts.min()
    print(f"\nBalancing to {min_count} samples per class.")

    out = []
    for lab, grp in df.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=42)
        out.append(grp.head(min_count))
    return pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

def sample_old_rows_per_class(df, frac=0.2):
    out = []
    for lab, grp in df.groupby("label"):
        n = max(1, int(len(grp) * frac))
        n = min(n, len(grp))
        out.append(grp.sample(n=n, random_state=42))
    if not out:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(out, ignore_index=True)


rows = []

# GoEmotions only (includes disgust/disapproval signals for the disgust class).
_, _, ds = safe_load([("go_emotions", "raw"), ("go_emotions", None)])
if ds:
    rows.extend(collect_rows_goemotions(ds))
else:
    raise RuntimeError("Could not load GoEmotions dataset.")

df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"])

df = df.drop_duplicates(subset=["text", "label"])
df = df[df["label"].isin(TARGET_LABELS)]

print("\nBefore balancing:")
print(df["label"].value_counts())
print("\nSource mix before balancing:")
print(df["source"].value_counts())

# Require all target classes before strict balancing.
present = set(df["label"].unique().tolist())
missing = [label for label in TARGET_LABELS if label not in present]
if missing:
    raise ValueError(
        f"Missing classes before balancing: {missing}. "
        "Cannot build a fully balanced dataset until these labels exist in sources."
    )

df = balance_per_class(df)

print("\nAfter balancing:")
print(df["label"].value_counts())
print(f"\nTotal rows: {len(df)}")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv("text_emotion_train.csv", index=False)
val_df.to_csv("text_emotion_val.csv", index=False)
df.to_csv("text_emotion_all.csv", index=False)

print("\nSaved:")
print(" - text_emotion_train.csv")
print(" - text_emotion_val.csv")
print(" - text_emotion_all.csv")
