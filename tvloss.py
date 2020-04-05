import torch 
import torch.nn as nn



class TVLossSegment(nn.Module):
    def __init__(self,TVLoss_weight=0.00001):
        super(TVLossSegment,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        #Ignored line segmentation in calculations
        h_tv = (torch.abs(x[:,0,1:,:]-x[:,0,:-1,:])).sum()+(torch.abs(x[:,2,1:,:]-x[:,2,:-1,:])).sum()
        w_tv = (torch.abs(x[:,0,:,1:]-x[:,0,:,:-1])).sum()+(torch.abs(x[:,2,:,1:]-x[:,2,:,:-1])).sum()
        return self.TVLoss_weight*(h_tv+w_tv)/batch_size

    def _tensor_size(self,t):
        return (t.size()[1]-1)*t.size()[2]*t.size()[3]

class TVLossDetect(nn.Module):
    def __init__(self,TVLoss_weight=0.00001):
        super(TVLossDetect,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        #Ignored line segmentation in calculations
        h_tv = (torch.abs(x[:,:,1:,:]-x[:,:,:-1,:])).sum()
        w_tv = (torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])).sum()
        return self.TVLoss_weight*(h_tv+w_tv)/batch_size

    def _tensor_size(self,t):
        return (t.size()[1]-1)*t.size()[2]*t.size()[3]