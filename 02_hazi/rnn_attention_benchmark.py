"""
Deep Learning Extension: BiLSTM with Attention vs. Classical TF-IDF Models
==========================================================================
Compares Multinomial Naive Bayes and Logistic Regression against sequential
Recurrent Neural Networks (LSTM and Bidirectional LSTM with Attention) on the
official UCI SMS Spam Collection dataset, measuring both predictive performance
and computational training/inference time.

Generates Figure 5 (fig5_rnn_attention_benchmark.png & .pdf) in Scientific English.
"""

import os
import time
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_FILE = os.path.join("data", "SMSSpamCollection")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. TEXT TOKENIZATION & PYTORCH DATASET FOR SEQUENTIAL MODELS
# ---------------------------------------------------------------------------
def simple_tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def build_vocab(texts, max_vocab=5000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=50):
    tokens = simple_tokenize(text)[:max_len]
    seq = [vocab.get(tok, 1) for tok in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq, tokens

class SMSSpamDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.sequences = [encode_text(t, vocab, max_len)[0] for t in texts]
        self.labels = labels.values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# ---------------------------------------------------------------------------
# 2. RECURRENT NEURAL NETWORK ARCHITECTURES (LSTM & BiLSTM + Attention)
# ---------------------------------------------------------------------------
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attn_weights_layer = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        mask = (x != 0).unsqueeze(-1)  # [batch_size, seq_len, 1]
        emb = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(emb)   # [batch_size, seq_len, hidden_dim*2]

        if self.use_attention:
            scores = self.attn_weights_layer(torch.tanh(lstm_out))  # [batch, seq_len, 1]
            scores = scores.masked_fill(~mask, -1e9)
            attn_weights = torch.softmax(scores, dim=1)             # [batch, seq_len, 1]
            context = torch.sum(lstm_out * attn_weights, dim=1)     # [batch, hidden_dim*2]
            logits = self.fc(self.dropout(context)).squeeze(-1)
            return logits, attn_weights.squeeze(-1)
        else:
            # Mean pooling over non-padded tokens for standard BiLSTM
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = (lstm_out * mask).sum(dim=1) / lengths
            logits = self.fc(self.dropout(pooled)).squeeze(-1)
            return logits, None

def train_and_eval_rnn(model, train_loader, test_loader, epochs=8, lr=0.003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Pos_weight to handle class imbalance slightly or standard BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0

    # Evaluation
    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            logits, _ = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))

    return np.array(all_preds), np.array(all_probs), train_time

# ---------------------------------------------------------------------------
# 3. MAIN COMPARISON & FIGURE 5 GENERATION
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Please run uci_spam_benchmark.py first so '{DATA_FILE}' is downloaded.")

    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "message"], quoting=3)
    X = df["message"]
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    # A) Classical TF-IDF Models
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_features=5000, sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    records = []

    # 1. Multinomial Naive Bayes
    t0 = time.perf_counter()
    mnb = MultinomialNB(alpha=0.2)
    mnb.fit(X_train_tfidf, y_train)
    mnb_time = time.perf_counter() - t0
    mnb_prob = mnb.predict_proba(X_test_tfidf)[:, 1]
    mnb_pred = (mnb_prob >= 0.5).astype(int)

    # 2. Logistic Regression
    t0 = time.perf_counter()
    lr = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
    lr.fit(X_train_tfidf, y_train)
    lr_time = time.perf_counter() - t0
    lr_prob = lr.predict_proba(X_test_tfidf)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)

    # B) Deep Learning Sequential Models (BiLSTM & BiLSTM + Attention)
    print("[INFO] Training PyTorch Sequential Models (BiLSTM & BiLSTM-Attention)...")
    vocab = build_vocab(X_train, max_vocab=5000, min_freq=2)
    max_len = 50
    train_ds = SMSSpamDataset(X_train, y_train, vocab, max_len=max_len)
    test_ds = SMSSpamDataset(X_test, y_test, vocab, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 3. Standard BiLSTM (Mean Pooling)
    torch.manual_seed(SEED)
    bilstm_model = BiLSTMAttentionClassifier(len(vocab), embed_dim=64, hidden_dim=64, use_attention=False)
    lstm_pred, lstm_prob, lstm_time = train_and_eval_rnn(bilstm_model, train_loader, test_loader, epochs=8)

    # 4. BiLSTM + Self-Attention
    torch.manual_seed(SEED)
    attn_model = BiLSTMAttentionClassifier(len(vocab), embed_dim=64, hidden_dim=64, use_attention=True)
    attn_pred, attn_prob, attn_time = train_and_eval_rnn(attn_model, train_loader, test_loader, epochs=8)

    model_outputs = [
        ("Multinomial NB", mnb_pred, mnb_prob, mnb_time),
        ("Logistic Regression", lr_pred, lr_prob, lr_time),
        ("BiLSTM (Pool)", lstm_pred, lstm_prob, lstm_time),
        ("BiLSTM + Attention", attn_pred, attn_prob, attn_time)
    ]

    for name, pred, prob, t_sec in model_outputs:
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        records.append({
            "Model": name,
            "Accuracy (%)": accuracy_score(y_test, pred) * 100,
            "Precision (%)": precision_score(y_test, pred) * 100,
            "Recall (%)": recall_score(y_test, pred) * 100,
            "F1-Score (%)": f1_score(y_test, pred) * 100,
            "PR-AUC": average_precision_score(y_test, prob),
            "FP": fp,
            "FN": fn,
            "Train Time (s)": t_sec
        })

    res_df = pd.DataFrame(records)
    print("\n--- CLASSICAL VS. DEEP LEARNING BENCHMARK SUMMARY ---")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    res_df.to_csv(os.path.join("data", "dl_vs_classical_benchmark.csv"), index=False)

    # -----------------------------------------------------------------------
    # FIGURE 5: 2-Panel Scientific Figure (Benchmark Trade-off + Attention Map)
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.9,
        "figure.dpi": 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A: Grouped Bar (F1 & Precision) + Log Training Time Line
    x = np.arange(len(res_df))
    width = 0.34
    b1 = ax1.bar(x - width/2, res_df["F1-Score (%)"], width, label="F1-Score (%)", color="#1f77b4", edgecolor="#103c5e")
    b2 = ax1.bar(x + width/2, res_df["Precision (%)"], width, label="Precision (%)", color="#2ca02c", edgecolor="#175217")
    ax1.set_ylim([85, 102])
    ax1.set_ylabel("Classification Metric (%)", fontweight="bold", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(res_df["Model"], fontweight="bold", rotation=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1_t = ax1.twinx()
    line_t = ax1_t.plot(x, res_df["Train Time (s)"], color="#d62728", marker="o", lw=2.2,
                        markersize=7, linestyle="--", label="Training Time (s, log scale)")
    ax1_t.set_yscale("log")
    ax1_t.set_ylabel("Training Time (seconds, log scale)", fontweight="bold", color="#d62728")
    ax1_t.set_ylim([1e-4, max(res_df["Train Time (s)"]) * 8])

    for i, t_val in enumerate(res_df["Train Time (s)"]):
        ax1_t.text(x[i], t_val * 2.2, f"{t_val:.3f}s" if t_val < 0.1 else f"{t_val:.1f}s",
                   color="#b22222", fontweight="bold", fontsize=8.5, ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, fontsize=8.5)
    ax1.set_title("A) Predictive Performance vs. Computational Cost", fontweight="bold", pad=10)

    # Panel B: Self-Attention Weights on a Representative Spam Message
    sample_spam = "URGENT! You have won a 1000 prize! Call 09061701461 to claim your free reward now!"
    seq, tokens = encode_text(sample_spam, vocab, max_len=len(simple_tokenize(sample_spam)))
    device = next(attn_model.parameters()).device
    with torch.no_grad():
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        logit, attn_w = attn_model(inp)
        prob_spam = torch.sigmoid(logit).item()
        weights = attn_w[0, :len(tokens)].cpu().numpy()

    # Plot horizontal attention bar / heatmap for tokens
    y_pos = np.arange(len(tokens))
    norm_w = weights / (weights.max() + 1e-9)
    colors_attn = plt.cm.YlOrRd(0.25 + 0.75 * norm_w)

    bars_attn = ax2.barh(y_pos, weights, color=colors_attn, edgecolor="#555555", height=0.72)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tokens, fontweight="bold", fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Self-Attention Weight (α_t)", fontweight="bold")
    ax2.set_title(f"B) BiLSTM-Attention Token Importance (P(Spam) = {prob_spam:.4f})", fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, w in zip(bars_attn, weights):
        ax2.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                 va="center", ha="left", fontsize=8, fontweight="bold", color="#333333")
    ax2.set_xlim(0, max(weights) * 1.22)

    plt.suptitle("Classical TF-IDF Classifiers vs. Recurrent Neural Networks (BiLSTM-Attention)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_rnn_attention_benchmark.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig5_rnn_attention_benchmark.pdf"), bbox_inches="tight")
    plt.close()
    print("\n[SUCCESS] Saved 'figures/fig5_rnn_attention_benchmark.png' (and .pdf)!")

if __name__ == "__main__":
    main()