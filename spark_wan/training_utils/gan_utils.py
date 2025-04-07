import torch
import torch.nn.functional as F


def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    if nll_loss == 0:
        return torch.tensor(1)

    d_weight_final = 0
    for layer in last_layer:
        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads.float()) / (torch.norm(g_grads.float()) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e6).detach()
        d_weight_final = d_weight_final + d_weight
    d_weight_final = d_weight_final / len(last_layer)
    return d_weight_final


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def d_loss(logits_real, logits_fake):
    loss_real = - torch.log(torch.sigmoid(logits_real.float()) + 1e-6).mean()
    loss_fake = - torch.log(1 - torch.sigmoid(logits_fake.float()) + 1e-6).mean()
    return loss_real + loss_fake

