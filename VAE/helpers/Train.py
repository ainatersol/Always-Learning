

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.regression import R2Score


def train_loop_vae(model, optimizer, loss_fn, train_loader, CFG, DEVICE='cpu', wandb=False, verbose=False):

    for epoch in range(CFG.NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            x = x.to(DEVICE) #.view(x.shape[0], INPUT_DIM)
            
            x_reconstructed, mu, sigma, z = model(x)
            
            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            
            log_var = sigma
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            ELBO = reconstruction_loss + CFG.beta*kl_div
            
            # Backprop
            loss = ELBO
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item)

            if wandb:
                wandb.log({'Loss':ELBO.item()})
        

        if (epoch%15 == 0) & (verbose == True):
            print(y[1])
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(x[1, 0, :, :].cpu().squeeze())
            ax[1].imshow(x_reconstructed[1, 0, :, :].cpu().detach().numpy().squeeze())
            plt.show()
            
            print(y[5])
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(x[5, 0, :, :].cpu().squeeze())
            ax[1].imshow(x_reconstructed[5, 0, :, :].cpu().detach().numpy().squeeze())
            plt.show()

        print(f"{epoch=}")
        print(f"{loss=}")

    return model

def train_loop_svae(model, optimizer, loss_fn, train_loader, CFG, DEVICE='cpu', wandb=False, verbose=False, th=0.5, use_clip_grad=False):
    scaler = GradScaler()
    for epoch in range(CFG.NUM_EPOCHS):
        y_pred_vec = []
        y_vec = []
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            x = x.to(DEVICE)
            y = (y > th).float().to(DEVICE)
            
            with autocast():
                x_reconstructed, mu, sigma, z, y_pred = model(x)
                
                # Compute loss
                y_pred_loss = loss_fn(y_pred.squeeze(), y)
                reconstruction_loss = loss_fn(x_reconstructed, x)
                
                log_var = sigma
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                ELBO = reconstruction_loss + CFG.beta*kl_div + CFG.alpha*y_pred_loss
            
            y_pred = y_pred.squeeze()
            y_pred_vec.append((y_pred>th))
            y_vec.append(y)

            # Backprop
            loss = ELBO
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item)
            if wandb:
                wandb.log({'Loss':ELBO.item()})

        if (epoch%15 == 0) & (verbose == True):
            print(y[1])
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(x[1, 0, :, :].cpu().squeeze())
            ax[1].imshow(x_reconstructed[1, 0, :, :].cpu().detach().numpy().squeeze())
            plt.show()
            
            print(y[5])
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(x[5, 0, :, :].cpu().squeeze())
            ax[1].imshow(x_reconstructed[5, 0, :, :].cpu().detach().numpy().squeeze())
            plt.show()
            
        y_pred_vec = torch.cat(y_pred_vec) # Concatenating lists into tensors
        y_vec = torch.cat(y_vec)
        correct = (y_pred_vec == y_vec).sum().item()
        acc = correct / len(y_vec)

        print(f"{acc=}")
        print(f"{epoch=}")
        print(f"{loss=}")

    return model

def train_loop_cls(model, optimizer, loss_fn, train_loader, CFG, DEVICE='cpu', wandb=False, verbose=False, th=0.5, use_clip_grad=False):

    scaler = GradScaler()
    for epoch in range(CFG.NUM_EPOCHS):
        y_pred_vec = []
        y_vec = []
        loop = tqdm(enumerate(train_loader))
        for batch, (x, y) in loop:
            x = x.to(DEVICE) 
            y = (y > th).float().to(DEVICE)
            
            with autocast():
                _, y_pred = model(x)
                #compute loss
                # cls_loss = loss_fn(y_pred, y)
                cls_loss = loss_fn(y_pred.squeeze().float(), y.float())
            
            y_pred = y_pred.squeeze()
            y_pred_vec.append((y_pred>th))
            y_vec.append(y)
            
            #zero gradients + backprop
            optimizer.zero_grad()
  
            scaler.scale(cls_loss).backward()

            if use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=cls_loss.item)

            if wandb:
                wandb.log({'Loss':cls_loss.item()})

        if (epoch%15 == 0) & (verbose == True):
            print(y[1], y_pred[1])
            fig, ax = plt.subplots(1,1)
            ax[0].imshow(x[1, 0, :, :].cpu().squeeze())
            plt.show()
            
            print(y[5], y_pred[5])
            fig, ax = plt.subplots(1,1)
            ax[0].imshow(x[5, 0, :, :].cpu().squeeze())
            plt.show()

        y_pred_vec = torch.cat(y_pred_vec) # Concatenating lists into tensors
        y_vec = torch.cat(y_vec)
        correct = (y_pred_vec == y_vec).sum().item()
        acc = correct / len(y_vec)

        print(f"{acc=}")
        print(f"{epoch=}")
        print(f"{cls_loss=}")

        if wandb:
                wandb.log({'Accuracy':acc})

    return model

def train_loop_regression(model, optimizer, loss_fn, train_loader, CFG, DEVICE='cpu', wandb=False, verbose=False, use_clip_grad=False):
    scaler = GradScaler()
    for epoch in range(CFG.NUM_EPOCHS):
        y_pred_vec = []
        y_vec = []
        loop = tqdm(enumerate(train_loader))
        for batch, (x, y) in loop:
            x = x.to(DEVICE) 
            y = y.float().to(DEVICE)
            
            with autocast():
                _, y_pred = model(x)
                #compute loss
                cls_loss = loss_fn(y_pred.squeeze().float(), y.float())
            
            y_pred = y_pred.squeeze()
            y_pred_vec.append(y_pred)
            y_vec.append(y)
            
            #zero gradients + backprop
            optimizer.zero_grad()
  
            scaler.scale(cls_loss).backward()

            if use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=cls_loss.item)

            if wandb:
                wandb.log({'Loss':cls_loss.item()})

        if (epoch%15 == 0) & (verbose == True):
            print(y[1], y_pred[1])
            fig, ax = plt.subplots(1,1)
            ax[0].imshow(x[1, 0, :, :].cpu().squeeze())
            plt.show()
            
            print(y[5], y_pred[5])
            fig, ax = plt.subplots(1,1)
            ax[0].imshow(x[5, 0, :, :].cpu().squeeze())
            plt.show()

        y_pred_vec = torch.cat(y_pred_vec) # Concatenating lists into tensors
        y_vec = torch.cat(y_vec)
        res = torch.sum(torch.abs(y_pred_vec-y_vec)**2)
        r2 = R2Score().to(DEVICE)(y_pred_vec, y_vec)

        print(f"{res=}")
        print(f"{r2=}")
        print(f"{epoch=}")
        print(f"{cls_loss=}")

        if wandb:
                wandb.log({'R2':r2})

    return model

