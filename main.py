import test

import torch
from torchinfo import summary

import prepare_data
import train
from model import MyNet
from visualize import plot_results

device = torch.device("cuda")
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)

train_loader, val_loader, test_loader = prepare_data()

summary(
    model=MyNet(),
    input_size=(1, 3, 32, 32),
    col_names=["input_size", "output_size", "num_params", "trainable"],
)

nn = torch.nn
model = MyNet().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="max", factor=0.5, patience=5
)
train_results = train(
    model,
    loss_func,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    50,
    device,
)

test_result = test(model, loss_func, test_loader, device)
print(
    f"Test Loss: {test_result['test_loss']:.4f}, Test Accuracy: {test_result['test_acc']:.4f}"
)
plot_results(train_results)
