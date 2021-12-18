import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_step = 1000
epsilon = 8 / 255
alpha = epsilon/ n_step
# std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

def modelPerformance(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone()  # initialize x_adv as original benign image x
    x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y)  # calculate loss
    loss.backward()  # calculate gradient
    x_adv = x_adv + epsilon * x_adv.grad.detach().sign()
    return x_adv


def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=int(3 * n_step)):
    for p in model.parameters(): p.requires_grad = False
    x_adv = x.detach().clone()
    for _ in range(num_iter):
        x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x-epsilon)
    return x_adv
