class EWC:
    def __init__(self, model: nn.Module, dataloader: DataLoader, device=DEVICE):
        self.model = copy.deepcopy(model).to(device)
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        # compute diagonal Fisher info
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        self.model.eval()
        for x_batch, in dataloader:
            x_batch = x_batch.to(self.device)
            self.model.zero_grad()
            # use negative log-likelihood surrogate: here MSE loss as proxy
            out = self.model(x_batch)
            loss = nn.MSELoss()(out, x_batch)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2
        # normalize
        for n in self.fisher:
            self.fisher[n] = self.fisher[n] / len(dataloader)

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = (self.fisher[n] * (p - self.params[n])**2).sum()
                loss += _loss
        return loss