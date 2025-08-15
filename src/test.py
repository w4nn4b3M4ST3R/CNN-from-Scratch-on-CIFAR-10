import torch


def test(model, loss_func, test_loader, device):
    model.eval()
    with torch.inference_mode():
        test_loss = 0.0
        test_acc = 0.0
        for X, y in test_loader:
            X, y = X.to(device), y.long().to(device)
            y_hat = model(X)
            loss = loss_func(y_hat, y)
            test_loss += loss.item()
            test_acc += (y_hat.argmax(dim=1) == y).sum().item()

        test_acc /= len(test_loader.dataset)
        test_loss /= len(test_loader)

        print(
            f"[INFO] - Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}"
        )

        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
