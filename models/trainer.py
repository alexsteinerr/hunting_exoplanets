import torch
import torch.optim as optim

def weighted_mse(pred, target, w):
    return torch.mean(w * (pred - target)**2)

def train_model(model, X_train, y_train, err_train, epochs=10000, patience=500, lr=1e-3, weight_decay=1e-4, alpha_transit=10.0):
    eps = 1e-8
    w_iv = 1.0 / torch.clamp(err_train**2, min=eps)
    if torch.any(torch.isfinite(w_iv)):
        w_iv = torch.clamp(w_iv, max=torch.quantile(w_iv[torch.isfinite(w_iv)], 0.99))
    else:
        w_iv = torch.ones_like(w_iv)
    
    w_transit = 1.0 + alpha_transit * torch.clamp(1.0 - y_train, min=0.0)
    weights = (w_iv * w_transit).detach()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1000))

    best_loss = float("inf")
    wait = 0
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = weighted_mse(pred, y_train, weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        cur = loss.item()
        if cur + 1e-10 < best_loss:
            best_loss, wait = cur, 0
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | wMSE {cur:.8e} | best {best_loss:.8e}")
            
        if wait >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} (best wMSE {best_loss:.8e})")
            break

    model.load_state_dict(best_state)
    return best_loss