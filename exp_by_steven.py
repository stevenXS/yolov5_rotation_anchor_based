import torch
import numpy as np
import socket
import torchvision.models as models



def torch_test():
    hsize,wsize = 4,4
    yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
    grid = torch.stack((xv,yv),2).view(1,1,hsize,wsize,2)
    grid = grid.view(1, -1, 2) # size = (1,20*20,2)

    #
    expanded_strides = []
    expanded_strides.append(
        torch.zeros(1,grid.shape[1])
             .fill_(8)
    )

    #
    x_shifts = []
    y_shifts = []
    x_shifts.append(grid[:, :, 0])
    y_shifts.append(grid[:, :, 1])
    x_shifts_per_image = x_shifts[0] * expanded_strides[0]
    y_shifts_per_image = y_shifts[0] * expanded_strides[0]

    x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides[0])
            .unsqueeze(0)
            .repeat(1,1,1)
    )
    y_centers_per_image = (
        (y_shifts_per_image + 0.5 * expanded_strides[0])
        .unsqueeze(0)
        .repeat(1,1, 1)
    )
    print(x_centers_per_image)
    print(y_centers_per_image)

def socket_test():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('202.115.52.9', 4370))
    s.send(b'614')
    data = s.recv(1024)
    print(s,data)

'''
tensor索引说明:
    1.bool型索引考虑这个维度中的数保留与否
    2.int型考虑的是这个维度中的数保留哪个
'''
def tensor_test():
    x = torch.randn(3,3,5,5)
    a = torch.tensor(2) # 取第2组[80,80,201]
    b = torch.tensor(2) # 取第2组[80,80,201]
    c = torch.tensor(2) # 取第2组[80,80,201]
    d = torch.tensor(2) # 取第2组[80,80,201]
    print(x[a,b,c,d]) # # 取第2组[80,80,201]，维度会变少，从左往右减少
    print(x[[1,1],[1,1]]) # 对于多个list的索引，可以把每个list看做一个坐标，[1,1]找第1组的第1个


if __name__=='__main__':
    # Model
    device = torch.device("cuda")
    x = torch.randn(1,3,224,224).to(device)
    x = x.clone().detach()
    # Inference
