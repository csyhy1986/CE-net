import torch
from dataset import Pair_set
from loss import clsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from CE_net import CE_net

EPOCH = 60
BATCH_SIZE = 32
LR = 0.001

train_set = Pair_set('train.dat')
train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, 
                        batch_size=BATCH_SIZE, drop_last=True)

net = CE_net()
net.cuda()
loss_cls = clsLoss().cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, EPOCH, eta_min=LR*0.01)

out_file = open('accuracy_cenet.dat', mode = 'w')
max_fr = 0.0
for epoch in range(1, EPOCH+1):
    mean_loss = []
    net.double().train()
    for i, data in enumerate(train_loader):
        x1 = data[1].cuda()
        x2 = data[2].cuda()
        lbls = data[3].cuda()
        w,_ = net(x1,x2)

        net.zero_grad()
        loss = loss_cls(w,lbls)
        loss.backward()
        optimizer.step()
        mean_loss.append(loss.detach())

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, EPOCH, i+1, 
                len(train_loader), torch.mean(torch.stack(mean_loss)).item()))    
            mean_loss = []
    scheduler.step()
    
    # if(f_r / len(eval_loader) > max_fr):
    torch.save(net.state_dict(), "model_weights.pth")
        # max_fr = f_r / len(eval_loader)
out_file.close()




