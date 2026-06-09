"""Model evaluation plots: ROC, PR, confusion matrices, loss curves, threshold analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc as sklearn_auc,
    average_precision_score, confusion_matrix
)

def plot_loss_curves(train_losses, val_losses, title="Training and Validation Loss"):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch'); plt.ylabel('BCE Loss'); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


def plot_roc_curves(trues, probs, class_names):
    n_classes = len(class_names)
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        if len(np.unique(trues[:, i])) == 2:
            fpr, tpr, _ = roc_curve(trues[:, i], probs[:, i])
            roc_auc = sklearn_auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    all_fprs = np.unique(np.concatenate([roc_curve(trues[:, i], probs[:, i])[0]
                                         for i in range(n_classes) if len(np.unique(trues[:, i]))==2]))
    mean_tpr = np.zeros_like(all_fprs)
    for i in range(n_classes):
        if len(np.unique(trues[:, i])) == 2:
            fpr, tpr, _ = roc_curve(trues[:, i], probs[:, i])
            mean_tpr += np.interp(all_fprs, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = sklearn_auc(all_fprs, mean_tpr)
    plt.plot(all_fprs, mean_tpr, '--', label=f'Macro-average (AUC = {macro_auc:.3f})', linewidth=2)
    plt.plot([0,1],[0,1],'k:', alpha=0.5)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Multi-label)')
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_precision_recall_curves(trues, probs, class_names):
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        if len(np.unique(trues[:, i])) == 2:
            precision, recall, _ = precision_recall_curve(trues[:, i], probs[:, i])
            ap = average_precision_score(trues[:, i], probs[:, i])
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_confusion_matrices(trues, bin_preds, class_names):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(10,4))
    if n_classes == 1: axes = [axes]
    for i, name in enumerate(class_names):
        cm = confusion_matrix(trues[:, i], bin_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No','Yes'], yticklabels=['No','Yes'])
        axes[i].set_title(name); axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
    plt.tight_layout(); plt.show()


def plot_threshold_analysis(trues, probs, class_names):
    thresholds = np.linspace(0.01, 0.99, 200)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(12,5))
    if n_classes == 1: axes = [axes]
    for i, name in enumerate(class_names):
        precisions, recalls, f1s = [], [], []
        for thresh in thresholds:
            pred = (probs[:, i] >= thresh).astype(int)
            tp = np.sum((trues[:, i]==1) & (pred==1))
            fp = np.sum((trues[:, i]==0) & (pred==1))
            fn = np.sum((trues[:, i]==1) & (pred==0))
            prec = tp/(tp+fp) if (tp+fp)>0 else 0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
        axes[i].plot(thresholds, precisions, label='Precision')
        axes[i].plot(thresholds, recalls, label='Recall')
        axes[i].plot(thresholds, f1s, label='F1', linewidth=2)
        best_idx = np.argmax(f1s)
        axes[i].axvline(x=thresholds[best_idx], color='gray', linestyle='--',
                         label=f"Best F1 thresh={thresholds[best_idx]:.2f}")
        axes[i].set_title(name); axes[i].set_xlabel('Threshold')
        axes[i].legend(loc='best'); axes[i].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_cv_macro_auc(cv_results):
    macro_aucs = [r['macro_AUC'] for r in cv_results]
    plt.figure(figsize=(6,5))
    plt.boxplot(macro_aucs, vert=True, patch_artist=True)
    plt.xticks([1], ['Macro AUC'])
    plt.ylabel('Macro AUC')
    plt.title(f'Cross-Validation Macro AUC\n(Mean={np.mean(macro_aucs):.3f} \u00b1 {np.std(macro_aucs):.3f})')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.show()
