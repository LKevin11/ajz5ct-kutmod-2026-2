"""
SMS Spam Classification Benchmark: Multinomial Naive Bayes vs. Logistic Regression
===================================================================================
This script automatically downloads the official UCI SMS Spam Collection dataset
(Almeida et al., 2011), extracts TF-IDF unigram and bigram features, evaluates
Multinomial Naive Bayes and L2-regularized Logistic Regression models (including
False Positive Rate analysis and threshold tuning), and exports publication-ready
scientific figures (300 DPI PNG & vector PDF) in English.

Authors: Kevin Kayasit Lendvay (AJZ5CT), Oliver Revesz
"""

import os
import io
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve
)

# ---------------------------------------------------------------------------
# 1. OFFICIAL UCI DATASET DOWNLOADER & LOADER
# ---------------------------------------------------------------------------
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = "data"
FIG_DIR = "figures"
RAW_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")

def load_uci_sms_dataset():
    """Downloads and loads the official UCI SMS Spam Collection dataset (N = 5,572)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(RAW_FILE):
        print(f"[INFO] Downloading official UCI SMS Spam Collection from:\n       {UCI_URL} ...")
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                z.extractall(DATA_DIR)
        print(f"[INFO] Successfully extracted dataset to '{RAW_FILE}'.")
    else:
        print(f"[INFO] Using cached official UCI dataset at '{RAW_FILE}'.")

    df = pd.read_csv(RAW_FILE, sep="\t", header=None, names=["label", "message"], quoting=3)
    # Also save a clean UTF-8 CSV copy for convenience
    df.to_csv(os.path.join(DATA_DIR, "uci_sms_spam_collection.csv"), index=False)
    return df

# ---------------------------------------------------------------------------
# 2. MAIN BENCHMARK & SCIENTIFIC PLOTTING PIPELINE
# ---------------------------------------------------------------------------
def main():
    df = load_uci_sms_dataset()
    n_total = len(df)
    n_ham = (df["label"] == "ham").sum()
    n_spam = (df["label"] == "spam").sum()
    print("=" * 75)
    print(f"DATASET SUMMARY: UCI SMS Spam Collection (Almeida et al., 2011)")
    print(f"Total messages : {n_total}")
    print(f"Legitimate (ham): {n_ham} ({n_ham / n_total * 100:.2f}%)")
    print(f"Spam           : {n_spam} ({n_spam / n_total * 100:.2f}%)")
    print("=" * 75)

    # Binary target encoding: 0 = Ham (Legitimate), 1 = Spam
    X = df["message"]
    y = (df["label"] == "spam").astype(int)

    # Stratified 75% / 25% Train-Test Split (reproducible with random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # TF-IDF Vectorization (Unigrams + Bigrams, sublinear TF scaling)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[INFO] TF-IDF Vocabulary Size: {X_train_tfidf.shape[1]} features (1-2 grams, min_df=2)")

    # Models:
    # 1) Multinomial Naive Bayes with Laplace smoothing (alpha=0.2 gives classic high recall / moderate FP trade-off)
    # 2) L2-regularized Logistic Regression (C=5.0, discriminative linear classifier)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    }

    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) * 100.0  # False Positive Rate (%)
        fnr = fn / (fn + tp) * 100.0  # False Negative Rate (%)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
            "TN": tn,
            "FP (Ham->Spam)": fp,
            "FN (Spam->Ham)": fn,
            "TP": tp,
            "FPR (%)": fpr,
            "FNR (%)": fnr
        })

    results_df = pd.DataFrame(results)
    print("\n--- TEST SET EVALUATION METRICS (N_test = {}) ---".format(len(y_test)))
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results_df.to_csv(os.path.join(DATA_DIR, "benchmark_metrics_summary.csv"), index=False)

    # -----------------------------------------------------------------------
    # SCIENTIFIC PLOTTING CONFIGURATION (IEEE / ACM Publication Style)
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.9,
        "figure.dpi": 300
    })

    # -----------------------------------------------------------------------
    # FIGURE 1: Precision-Recall Curve (Crucial for Imbalanced Spam Detection)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    colors = {"Multinomial Naive Bayes": "#1f77b4", "Logistic Regression": "#d62728"}
    styles = {"Multinomial Naive Bayes": "-", "Logistic Regression": "--"}

    for name in models.keys():
        prec, rec, _ = precision_recall_curve(y_test, probabilities[name])
        ap = average_precision_score(y_test, probabilities[name])
        ax.plot(
            rec, prec,
            label=f"{name} (AP = {ap:.4f})",
            color=colors[name],
            linestyle=styles[name],
            linewidth=2.2
        )

    baseline = y_test.mean()
    ax.axhline(
        y=baseline, color="#555555", linestyle=":", linewidth=1.3,
        label=f"Random Classifier Baseline (Prevalence = {baseline:.3f})"
    )

    ax.set_xlabel("Recall (Sensitivity / True Positive Rate)", fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax.set_title("Precision-Recall Curves on the UCI SMS Spam Test Set", fontweight="bold", pad=10)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.04])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_precision_recall_curve.png"), dpi=300)
    plt.savefig(os.path.join(FIG_DIR, "fig1_precision_recall_curve.pdf"))
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 2: Side-by-Side Confusion Matrices (Counts & Row Percentages)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0

        annot_labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_labels[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.2f}%)"

        sns.heatmap(
            cm, annot=annot_labels, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"],
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1.0, linecolor="#e0e0e0"
        )
        ax.set_title(name, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class Label", fontweight="bold")
        ax.set_ylabel("True Class Label", fontweight="bold")

    plt.suptitle(
        f"Confusion Matrices on the Hold-Out Test Set (N = {len(y_test)})",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig2_confusion_matrices.pdf"), bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 3: Top 15 Spam-Indicative Features (LR Coefficients & NB Log-Odds)
    # -----------------------------------------------------------------------
    feature_names = np.array(vectorizer.get_feature_names_out())
    lr_model = models["Logistic Regression"]
    nb_model = models["Multinomial Naive Bayes"]

    lr_coefs = lr_model.coef_[0]
    # Log-probability ratio for Naive Bayes: log P(w|Spam) - log P(w|Ham)
    nb_log_odds = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]

    top_k = 15
    top_lr_idx = np.argsort(lr_coefs)[-top_k:]
    top_nb_idx = np.argsort(nb_log_odds)[-top_k:]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # Panel A: Logistic Regression Weights
    ax = axes[0]
    bars1 = ax.barh(
        feature_names[top_lr_idx], lr_coefs[top_lr_idx],
        color="#b22222", edgecolor="#6b1414", height=0.68, alpha=0.9
    )
    ax.set_title("A) Logistic Regression Top Spam Weights (β)", fontweight="bold", pad=10)
    ax.set_xlabel("L2-Regularized Coefficient (β_j)", fontweight="bold")
    ax.set_ylabel("TF-IDF N-gram Feature", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color="#4a0e0e")
    ax.set_xlim(0, max(lr_coefs[top_lr_idx]) * 1.18)

    # Panel B: Multinomial Naive Bayes Log-Likelihood Ratio
    ax = axes[1]
    bars2 = ax.barh(
        feature_names[top_nb_idx], nb_log_odds[top_nb_idx],
        color="#1f5a8c", edgecolor="#10324f", height=0.68, alpha=0.9
    )
    ax.set_title("B) Naive Bayes Top Log-Likelihood Ratios", fontweight="bold", pad=10)
    ax.set_xlabel("Log-Odds Ratio: log P(w|Spam) - log P(w|Ham)", fontweight="bold")
    ax.set_ylabel("TF-IDF N-gram Feature", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color="#10324f")
    ax.set_xlim(0, max(nb_log_odds[top_nb_idx]) * 1.18)

    plt.suptitle("Top 15 Most Informative Spam Tokens Across Generative and Discriminative Models",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_top_spam_features.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig3_top_spam_features.pdf"), bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 4: Decision Threshold vs. False Positive Rate & Recall Trade-off
    # (Directly addresses the research question on misclassifying normal messages)
    # -----------------------------------------------------------------------
    thresholds = np.linspace(0.05, 0.95, 91)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=False)

    for ax, name in zip(axes, models.keys()):
        y_prob = probabilities[name]
        fprs, recalls, precs = [], [], []
        for t in thresholds:
            preds_t = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, preds_t).ravel()
            fprs.append(fp / (fp + tn) * 100.0)
            recalls.append(tp / (tp + fn) * 100.0)
            precs.append(precision_score(y_test, preds_t, zero_division=1.0) * 100.0)

        ax2 = ax.twinx()
        l1 = ax.plot(thresholds, recalls, color="#1f77b4", lw=2.0, label="Spam Recall (%)")
        l2 = ax.plot(thresholds, precs, color="#2ca02c", lw=2.0, linestyle="--", label="Spam Precision (%)")
        l3 = ax2.plot(thresholds, fprs, color="#d62728", lw=2.0, linestyle="-.", label="False Positive Rate (%)")

        ax.axvline(0.5, color="gray", linestyle=":", lw=1.2, label="Default Threshold (0.50)")
        ax.set_title(f"{name}: Threshold Sensitivity", fontweight="bold", pad=10)
        ax.set_xlabel("Classification Decision Threshold (τ)", fontweight="bold")
        ax.set_ylabel("Precision / Recall (%)", fontweight="bold", color="#1f77b4")
        ax2.set_ylabel("False Positive Rate - Ham as Spam (%)", fontweight="bold", color="#d62728")
        ax.set_ylim([50, 102])
        ax2.set_ylim([0, max(fprs) * 1.15])
        ax.grid(True, linestyle="--", alpha=0.35)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="center left", frameon=True, fontsize=8.5)

    plt.suptitle("Impact of Decision Threshold (τ) on False Positive Rate and Spam Detection",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_threshold_fpr_tradeoff.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig4_threshold_fpr_tradeoff.pdf"), bbox_inches="tight")
    plt.close()

    print("\n[SUCCESS] All scientific figures have been generated in the 'figures/' directory:")
    print("  1. figures/fig1_precision_recall_curve.png (and .pdf)")
    print("  2. figures/fig2_confusion_matrices.png     (and .pdf)")
    print("  3. figures/fig3_top_spam_features.png      (and .pdf)")
    print("  4. figures/fig4_threshold_fpr_tradeoff.png (and .pdf)")

if __name__ == "__main__":
    main()
