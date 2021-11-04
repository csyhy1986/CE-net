import numpy as np
import torch
from torch.utils.data import Dataset
import os

class Pair_set(Dataset):
    def __init__(self, datasets_base_path):
        super(Pair_set, self).__init__()
        self.data = []
        self.n_img_pairs = 0
        self.n_pt_pairs_per_img = 0
        data_lines = []
        if (os.path.exists(datasets_base_path)):
            with open(datasets_base_path, "r") as ifp:
                lines = ifp.readlines()
                line_one = lines[0].replace('\\n','').split()
                self.n_img_pairs, self.n_pt_pairs_per_img = int(line_one[0]), int(line_one[1])
                for i in range(1,(self.n_pt_pairs_per_img+1)*self.n_img_pairs+1):
                    line = lines[i].replace('\\n','').split()
                    data_lines.append(line)
        
        for i in range(self.n_img_pairs):
            data_dic = {}
            pair_data = data_lines[i*(self.n_pt_pairs_per_img+1): 
                                   (i+1)*(self.n_pt_pairs_per_img+1)]
            F = np.stack([float(num) for num in pair_data[0]])
            data_dic['F'] = F
            mpt1 = []
            mpt2 = []
            labels = []
            for j in range(1,self.n_pt_pairs_per_img+1):
                t_pair_data = pair_data[j][:4]
                mpts = [float(num) for num in t_pair_data]
                mpt1.append([mpts[0], mpts[1]])
                mpt2.append([mpts[2], mpts[3]])
                label = float(pair_data[j][4])
                labels.append(label)
            mpt1 = np.stack(mpt1)
            mpt2 = np.stack(mpt2)
            labels = np.stack(labels)
            data_dic['PTLs'] = mpt1
            data_dic['PTRs'] = mpt2
            data_dic['LBLs'] = labels
            self.data.append(data_dic)
            
    def __getitem__(self, index):
        F = self.data[index]['F']
        ptsL = torch.from_numpy(self.data[index]['PTLs'])
        ptsR = torch.from_numpy(self.data[index]['PTRs'])
        lbls = self.data[index]['LBLs']
        return (F, ptsL, ptsR, lbls)

    def __len__(self):
        return len(self.data)