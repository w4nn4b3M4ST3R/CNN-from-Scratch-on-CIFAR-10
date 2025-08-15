import torch
from tqdm import tqdm


def train(
    model,
    loss_func,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    epochs,
    device,
):
    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        for X, y in tqdm(train_loader):
            X, y = X.to(device), y.long().to(device)
            y_hat = model(X)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()

        with torch.inference_mode():
            val_loss = 0.0
            val_acc = 0.0
            for X, y in valid_loader:
                X, y = X.to(device), y.long().to(device)
                y_hat = model(X)
                loss = loss_func(y_hat, y)

                val_loss += loss.item()
                val_acc += (y_hat.argmax(dim=1) == y).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)
        train_acc /= len(train_loader.dataset)
        val_acc /= len(valid_loader.dataset)

        if scheduler is not None:
            scheduler.step(val_acc)

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["train_acc"].append(train_acc)
        results["val_acc"].append(val_acc)

        print(
            f"[INFO] - Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    return results
