"""
Train a two-tower retrieval model and save the best model.
"""

INPUT_DIM_QUERY = 32
INPUT_DIM_PRODUCT = 160

HIDDEN_DIM_PRODUCT = 64
HIDDEN_DIM_QUERY = 64

EMBEDDING_DIM = 32
LR = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_SPLITS = 5
MODEL_DIR = 'models/two_towers'




import os
import time
import torch
from torch import nn, autocast
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold
from recsys.recall.two_towers.model import TwoTowerModel
from recsys.recall.two_towers.utils import select_device
from torch.nn import functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


def create_dataloaders(qd, pd, lbl, train_idx, val_idx, batch_size=32):
    train_ds = TensorDataset(qd[train_idx], pd[train_idx], lbl[train_idx])
    val_ds   = TensorDataset(qd[val_idx],   pd[val_idx],   lbl[val_idx])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size)
    )


def point_wise_loss(model, q_batch, p_batch, y_batch):
    """Point-wise loss using cross entropy on cosine similarity"""
    out = model(q_batch, p_batch)
    return nn.BCEWithLogitsLoss()(out, y_batch)

def pair_wise_loss(model, q_batch, p_pos_batch, p_neg_batch, margin=0.1):
    """Pair-wise loss with margin"""
    pos_sim = model(q_batch, p_pos_batch)
    neg_sim = model(q_batch, p_neg_batch)
    loss = torch.clamp(margin - (pos_sim - neg_sim), min=0)
    return loss.mean()

def list_wise_loss(model, q_batch, p_batch, r_batch, log_probs):
    """List-wise loss using softmax and cross entropy"""
    # Calculate cosine similarities
    sim = model(q_batch, p_batch)
    
    # Subtract log probabilities from similarities
    sim = sim - log_probs
    
    # Apply softmax to get probabilities
    probs = torch.softmax(sim, dim=1)
    
    # Calculate cross entropy loss weighted by rewards
    loss = -torch.sum(r_batch * torch.log(probs + 1e-8))
    return loss

def train_one_epoch(model, loader, loss_fn, optimizer, device, training_method="point_wise", log_probs=None):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        if training_method == "point_wise":
            q_batch, p_batch, y_batch = [b.to(device) for b in batch]
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            loss = point_wise_loss(model, q_batch, p_batch, y_batch)
            
        elif training_method == "pair_wise":
            q_batch, p_pos_batch, p_neg_batch, y_batch = [b.to(device) for b in batch]
            loss = pair_wise_loss(model, q_batch, p_pos_batch, p_neg_batch)
            
        elif training_method == "list_wise":
            q_batch, p_batch, r_batch = [b.to(device) for b in batch]
            loss = list_wise_loss(model, q_batch, p_batch, r_batch, log_probs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * q_batch.size(0)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, loss_fn, device, training_method="point_wise", log_probs=None):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            if training_method == "point_wise":
                q_batch, p_batch, y_batch = [b.to(device) for b in batch]
                if y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(1)
                loss = point_wise_loss(model, q_batch, p_batch, y_batch)
                
            elif training_method == "pair_wise":
                q_batch, p_pos_batch, p_neg_batch, y_batch = [b.to(device) for b in batch]
                loss = pair_wise_loss(model, q_batch, p_pos_batch, p_neg_batch)
                
            elif training_method == "list_wise":
                q_batch, p_batch, r_batch = [b.to(device) for b in batch]
                loss = list_wise_loss(model, q_batch, p_batch, r_batch, log_probs)
            
            total_loss += loss.item() * q_batch.size(0)
    
    return total_loss / len(loader.dataset)

def plot_losses(train_losses, val_losses, epoch, num_epochs):
    """Plot training and validation losses in real-time"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Loss vs Epoch (Current Epoch: {epoch}/{num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, len(train_losses), max(1, len(train_losses)//10)))
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def run_cross_validation(qd, pd, lbl, n_splits=5, batch_size=32, num_epochs=100, lr=1e-3, 
                        model_dir='models', training_method="point_wise"):
    device = select_device()
    print(f"Training on {device}\nCV folds={n_splits}, epochs={num_epochs}, lr={lr}\n")
    
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create subdirectory for the specific training method
    method_dir = os.path.join(model_dir, training_method)
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)
    
    qd, pd, lbl = qd.float(), pd.float(), lbl.float()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_loss, best_path = float('inf'), ''
    
    # Initialize plot
    plt.ion()
    train_losses = []
    val_losses = []
    
    # Calculate log probabilities for list-wise training
    log_probs = None
    if training_method == "list_wise":
        product_counts = pd.value_counts()
        probs = product_counts / product_counts.sum()
        log_probs = torch.tensor(np.log(probs.values), device=device)
    
    for fold, (tr_i, val_i) in enumerate(kf.split(qd), 1):
        train_loader, val_loader = create_dataloaders(qd, pd, lbl, tr_i, val_i, batch_size)
        model = TwoTowerModel(qd.size(1), pd.size(1), 64, 64, 32).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        
        # Reset losses for new fold
        train_losses = []
        val_losses = []
        
        for epoch in range(1, num_epochs+1):
            start = time.time()
            tl = train_one_epoch(model, train_loader, None, optimizer, device, 
                               training_method, log_probs)
            vl = evaluate(model, val_loader, None, device, 
                        training_method, log_probs)
            elapsed = time.time() - start
            
            # Store losses
            train_losses.append(tl)
            val_losses.append(vl)
            
            # Update plot
            plot_losses(train_losses, val_losses, epoch, num_epochs)
            
            if epoch % 10 == 0 or epoch == num_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Fold {fold}, Ep {epoch}/{num_epochs}: TL={tl:.4f}, VL={vl:.4f}, '
                      f'LR={current_lr:.2e}, time={elapsed:.2f}s')
        
        if vl < best_loss:
            best_loss, best_path = vl, os.path.join(method_dir, f'best_fold{fold}.pt')
            torch.save(model.state_dict(), best_path)
    
    plt.ioff()
    plt.close()
    return best_path

def main(train_inputs, train_labels, val_inputs, val_labels, training_method="point_wise"):
    device = select_device()
    
    if training_method == "point_wise":
        train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
        train_p = torch.tensor(train_inputs[1], dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        val_q = torch.tensor(val_inputs[0], dtype=torch.float32)
        val_p = torch.tensor(val_inputs[1], dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.float32)
        
    elif training_method == "pair_wise":
        train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
        train_p_pos = torch.tensor(train_inputs[1], dtype=torch.float32)
        train_p_neg = torch.tensor(train_inputs[2], dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        val_q = torch.tensor(val_inputs[0], dtype=torch.float32)
        val_p_pos = torch.tensor(val_inputs[1], dtype=torch.float32)
        val_p_neg = torch.tensor(val_inputs[2], dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.float32)
        
    elif training_method == "list_wise":
        train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
        train_p = torch.tensor(train_inputs[1], dtype=torch.float32)
        train_r = torch.tensor(train_inputs[2], dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        val_q = torch.tensor(val_inputs[0], dtype=torch.float32)
        val_p = torch.tensor(val_inputs[1], dtype=torch.float32)
        val_r = torch.tensor(val_inputs[2], dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.float32)
    
    best = run_cross_validation(
        train_q, train_p, train_y,
        n_splits=NUM_SPLITS,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        model_dir=MODEL_DIR,
        training_method=training_method
    )
    
    model = TwoTowerModel(
        q_in=INPUT_DIM_QUERY,
        p_in=INPUT_DIM_PRODUCT,
        q_hidden=HIDDEN_DIM_QUERY,
        p_hidden=HIDDEN_DIM_PRODUCT,
        emb=EMBEDDING_DIM
    ).to(device)
    
    model.load_state_dict(torch.load(best, map_location=device))
    test_loader = DataLoader(TensorDataset(val_q, val_p, val_y), batch_size=BATCH_SIZE)
    test_loss = evaluate(model, test_loader, None, device, training_method)
    print(f'Final Test Loss: {test_loss:.4f}')
    
    # Save final model with training method in the name
    final_model_path = os.path.join(MODEL_DIR, training_method, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)

if __name__ == '__main__':
    from recsys.data.load_data import load_data
    method = "pair_wise"
    train_inputs, train_labels, val_inputs, val_labels = load_data(training_method=method)
    print(f"Training method: {method}")
    main(train_inputs, train_labels, val_inputs, val_labels, training_method=method)
