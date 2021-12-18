import utilities.utils as utils
import torch
import torch.nn as nn

class dynamicWeightedCELoss(nn.Module):
    
    def __init__(self):
        super(dynamicWeightedCELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, outputs, targets):
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets[:,0].unsqueeze(-1), 1)
        
        subtargets_onehot = torch.zeros_like(outputs)
        subtargets_onehot.zero_()
        subtargets_onehot.scatter_(1, targets[:,1].unsqueeze(-1), 1)
        weight = torch.ones(subtargets_onehot.shape).to(subtargets_onehot.get_device()) + 2 * subtargets_onehot
        
        # outputs = self.softmax(outputs)
        loss = -targets_onehot.float() * torch.log_softmax(outputs,dim = 1)
        # return torch.mean(weight * loss)
        return torch.sum(weight * loss)/outputs.shape[0]

class softlabelCELoss(nn.Module):
    def __init__(self):
        super(softlabelCELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, outputs, targets, weight:list):
        target = torch.zeros_like(outputs)
        target.zero_()
        for t, w in zip(targets[:], weight):
            tmp = torch.zeros_like(outputs)
            tmp.zero_()
            tmp.scatter_(1, t.unsqueeze(-1), 1)
            target += w * tmp
        loss = -target.float() * torch.log_softmax(outputs,dim = 1)
        return torch.sum(loss)/outputs.shape[0]
class customCELoss(nn.Module):
    def __init__(self, eps: float, srng: utils.SRNG, dataset):
        super(customCELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dataset = dataset
        self.eps = eps
        self.srng = srng
        self.mse = nn.MSELoss()

    def updateSRNG(self, srng: utils.SRNG):
        self.srng = srng

    def forward(self, outputs, targets, test=False):
        targets_onehot = torch.zeros_like(outputs).zero_().to("cuda")
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)
        outputs = self.softmax(outputs)
        loss = -targets_onehot.float() * torch.log(outputs)

        if not test:
            highNoiseLabels = [self.dataset[self.srng.gen()][1]
                               for _ in range(outputs.shape[0])]
            highNoiseLabels = torch.tensor(highNoiseLabels).to("cuda")

            highNoise_onehot = torch.zeros_like(outputs).zero_().to("cuda")
            highNoise_onehot.scatter_(1, highNoiseLabels.unsqueeze(-1), 1)
            loss = loss - self.eps * \
                self.mse(outputs, highNoise_onehot.float())

        return torch.mean(loss)

    def test(self):
        return self.srng.gen()