import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, forecast_model, norm_model):
        super().__init__()
        self.fm = forecast_model
        self.nm = norm_model

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def forward(self, batch_x, labels):

        norm_x, pre_sta = self.nm(batch_x, 'norm') # CAN 
        
        forecast = self.fm(norm_x) # 
    
        forecast = self.nm(forecast, 'denorm')
        
        return forecast,1