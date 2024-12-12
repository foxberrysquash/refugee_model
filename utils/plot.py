
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score

import torch
## Plotting functions

def plot_loss(loss, figure_size = (10,6), ylabel = "Loss", title = "Training Loss over Epochs"):
    plt.figure(figsize=figure_size)
    sns.lineplot(x=range(len(loss)), y=loss, color= "b")

    # Customize the plot
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
    

def plot_roc(predictions,labels,average_precision = True, color = "red"):
    # Convert lists to tensors
    if not isinstance(predictions,torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(labels,torch.Tensor):
        labels = torch.tensor(labels)
    
    # Step 3: ROC Curve and AUC
    fpr, tpr, _ = roc_curve(labels.cpu(), predictions.cpu())
    roc_auc = auc(fpr, tpr)
    
    title_part = ""
    out = roc_auc
    
    if average_precision:
        # Average precision
        ap = average_precision_score(labels.cpu(), predictions.cpu())
        title_part = f" and AP = {ap:.3f}"
        out = roc_auc,ap
    # Plot ROC Curve using seaborn
    sns.set(style="whitegrid")
    plt.figure()
    sns.lineplot(x=fpr, y=tpr, color=color, lw=2, label=f'ROC curve (area = {roc_auc:.2f})', errorbar=None)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC with AUC = {roc_auc:.3f}" + title_part)
    plt.legend(loc='lower right')
    plt.show()
    
    return out