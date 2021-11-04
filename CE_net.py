# NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences (CVPR 2019 oral)
# This repository is a reference implementation for Chen Zhao, Zhiguo Cao, Chi Li, Xin Li, and Jiaqi Yang, 
# "NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences", CVPR 2019 oral. 
# If you use this code in your research, please cite the paper.
import torch
import torch.nn as nn
from torch.nn import functional as F
from attention_gnn import AttentionalGNN

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1]   # (batch_size, num_points, k)

    return idx[:, :, 1:]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)   # (batch_size, num_points, k)
    else:
        idx_out = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)  
        out = out + x1
        return F.relu(out)


class NM_block(nn.Module):
    def __init__(self, k_n):
        super(NM_block, self).__init__()
        self.k_n = k_n
        self.conv0 = nn.Sequential(
            nn.Conv2d((128+1)*2, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Conv2d(128, 1, (1, 1))

        self.res1 = ResNet_Block(128, 128, pre=False)
        self.res2 = ResNet_Block(128, 128, pre=False)
        self.res3 = ResNet_Block(128, 128, pre=False)
        self.res4 = ResNet_Block(128, 128, pre=False)

    def graph_feature(self, x):
        out = x
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-5)
        A = torch.bmm(out.permute(0, 2, 1).contiguous(), out)

        eye = torch.eye(A.size(1)).unsqueeze(0).repeat(A.size(0), 1, 1).cuda()
        D = torch.sum(A - eye, dim=1, keepdim=True) / A.size(1)
        out = torch.cat([x.unsqueeze(-1), torch.relu(torch.tanh(D.unsqueeze(-1)))], dim=1)
        
        return out

        
    def forward(self, data, x):
        out = self.graph_feature(data)
        
        idx = knn(out.squeeze(-1).contiguous(), k=self.k_n)
        out = get_graph_feature(out, k=self.k_n, idx=idx)
        out = self.conv0(out)
        out = F.max_pool2d(out, (1, self.k_n))

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        logit = self.linear(out)
        logit = logit.view(logit.size(0), -1)
        return logit

class NM_net(nn.Module):
    def __init__(self):
        super(NM_net, self).__init__()
        self.block = NM_block(k_n=8)

    def forward(self, x):
        w = self.block(x, x)
        return w

class CE_net(nn.Module):
    def __init__(self):
        super(CE_net, self).__init__()
        self.atn = AttentionalGNN(64,['self', 'self']*4)
        self.expand = nn.Sequential(nn.Conv2d(2, 64, (1, 1)),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, (1, 1)))
        self.nnm = NM_net()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, x1, x2):
        a1 = self.expand(x1.transpose(2,1).unsqueeze(-1))
        a2 = self.expand(x2.transpose(2,1).unsqueeze(-1))
        descs1, descs2, prob = self.atn(a1.squeeze(-1),a2.squeeze(-1))
        x = torch.cat((descs1,descs2),dim=1)
        return self.nnm(x), prob





# model = CE_Net()
# model.eval()
# model.cuda()
# x = torch.rand(32, 200, 4).cuda()  # Normalized matches: Batch_size * N *4
# weights, Es = model(x)       # Lists of predicted weights and Es 
# weights = weights[-1][-1]    # Batch_size * N
# Es = Es[-1]                  # Batch_size * 9
# mask = weights > 0
# a = 0