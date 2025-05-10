import torch
import torch.nn as nn

class CAN_ST(nn.Module):
    def __init__(self, num_nodes: int, clusters: int):
        super(CAN_ST, self).__init__()
        self.num_features = num_nodes
        self.eps = 1e-5
        self.affine = False
        self.len = 12
        self.clusters = clusters

        self._init_params()

    def forward(self, x, mode:str):
        b, t, n, d = x.shape
        x_else = x[:,:,:,1:]
        x = x[:,:,:,0]
              

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
            x = torch.cat((x.reshape(b, t, n, 1), x_else), dim=-1)
        
            return x, torch.cat([self.y_mean, self.y_std], dim = -1)
        elif mode == 'denorm':
            x = self._denormalize(x).unsqueeze(-1)
            return x
        else: raise NotImplementedError
        

    def _init_params(self):
        self.cluster_weight = nn.Parameter(torch.randn(self.len * self.num_features, self.clusters))
        
        self.mean_pool = nn.Parameter(torch.randn(self.clusters, 1))
        self.std_pool = nn.Parameter(torch.randn(self.clusters, 1))
    def _get_statistics(self, x):
        b, t, n = x.shape
        dim2reduce = tuple(range(1, x.ndim))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        

        self.cluster_score = torch.softmax((x.reshape(b, t * n)) @ self.cluster_weight, dim = 1)

        breakpoint()
        print(self.cluster_score)
        self.mean_affine = (self.cluster_score @ self.mean_pool).unsqueeze(-1)
        self.std_affine = (self.cluster_score @ self.std_pool).unsqueeze(-1)
        self.y_mean = ( torch.nn.functional.sigmoid(torch.bmm(self.mean_affine,self.mean)) + self.mean)
        self.y_std = (torch.nn.functional.sigmoid(torch.bmm(self.std_affine,self.stdev)) + self.stdev)


    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x



    def _denormalize(self, x):
        x = x * self.y_std
        x = x + self.y_mean
        return x
