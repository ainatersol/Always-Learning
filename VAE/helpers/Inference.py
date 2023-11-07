
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
)
import numpy as np

def inference_loop(model, image_loader, DEVICE = "cuda", v = False, mtype= 'vae') :    
    mu_vect = []
    y_vect = []
    images_recon = []
    z_vect = []
    model.eval()

    for batch, (x, y) in tqdm(enumerate(image_loader)):
        x = x.to(DEVICE) #.view(x.shape[0], INPUT_DIM)
        if mtype == 'vae':
            with torch.no_grad():
                x_recon, mu, sigma, z = model(x)
        elif mtype == 'svae':
            with torch.no_grad():
                x_recon, mu, sigma, z, y_pred = model(x)
        
        z_vect.append(z.cpu().detach().numpy())
        mu_vect.append(mu.cpu().detach().numpy())
        y_vect.append(y.cpu())
        images_recon.append(x_recon.cpu().detach().numpy())

        if v==True:
        
            print(y[1])
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(x[1, 0, :, :].cpu().squeeze())
            ax[1].imshow(x_recon[1, 0, :, :].cpu().detach().numpy().squeeze())
            plt.show()
            
            print(y[5])
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(x[5, 0, :, :].cpu().squeeze())
            ax[1].imshow(x_recon[5, 0, :, :].cpu().detach().numpy().squeeze())
            plt.show()
    
    X_vae_recons = np.concatenate(images_recon, axis=0)
    X_vae_recons = X_vae_recons.reshape(X_vae_recons.shape[0],-1)

    X_vae_z = np.concatenate(z_vect, axis=0)
    X_vae_z = X_vae_z.reshape(X_vae_z.shape[0],-1)

    y_vae = np.concatenate(y_vect, axis=0)

    return X_vae_recons, X_vae_z, y_vae

def inference_loop_cls(model, image_loader, DEVICE = "cuda"):
    X_pred_vec = []
    y_pred_vec = []
    y_vec = []
    model.eval()
    
    for batch, (x, y) in tqdm(enumerate(image_loader)):
        x = x.to(DEVICE) 
        y = y.to(DEVICE)
        with torch.no_grad():
            with autocast():
                    X_pred, y_pred = model(x)
        y_pred = y_pred.squeeze()
        X_pred_vec.append(X_pred)
        y_pred_vec.append((y_pred).float())
        y_vec.append(y)

    X_pred_vec = torch.cat(X_pred_vec).cpu().detach().numpy()
    y_pred_vec = torch.cat(y_pred_vec).cpu().detach().numpy()
    y_vec = torch.cat(y_vec).cpu().detach().numpy()

    return X_pred_vec, y_pred_vec, y_vec

def compute_metrics_cls_for_threshold(y_true, y_pred, threshold=0.5):
    y_true = np.where(y_true >= threshold, 1, 0)
    y_pred = np.where(y_pred >= threshold, 1, 0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    stats = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "ROC-AUC": roc_auc_score(
            y_true, y_pred
        ),  # Ensure that y_pred has probability estimates
        "Matthews Correlation Coefficient": matthews_corrcoef(y_true, y_pred),
    }

    # Print the results
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats