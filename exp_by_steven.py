import torch
import numpy as np
import socket
import torchvision.models as models
import cv2


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
    y = torch.randn(2,5)
    a = torch.tensor(2) # 取第2组[80,80,201]
    b = torch.tensor(2) # 取第2组[80,80,201]
    c = torch.tensor(2) # 取第2组[80,80,201]
    d = torch.tensor(2) # 取第2组[80,80,201]
    # print(x[a,b,c,d]) # # 取第2组[80,80,201]，维度会变少，从左往右减少
    # print(x[[1,1],[1,1]]) # 对于多个list的索引，可以把每个list看做一个坐标，[1,1]找第1组的第1个
    print(y)
    print(y[0])
    y[0, [0, 4, 4, 4, 2]] = 1.0
    print(y[0])

# 增加中心度计算 FCOS参考：https://blog.csdn.net/u014380165/article/details/90962991
def get_anchor_mask(self, pi, t, num_gt, layer_index):
        '''
        @param pi: 传入的是batch中某一个的特征图，size = (1种scale框 * size1 * size2, [x,y,w,h,theta,obj,classes])
            eg: 将检测头的每一层输出的特征图w，h进行合并
        @param targets: 当前batch中某一个image的targets，shape=(kinds_of_anchors, num_gt, feature_size1*feature_size2, [class,x,y,w,h,theta])
        '''

        strides = [8, 16, 32]  # 三层layer的采样率
        kinds_of_anchors = self.na  # 预测框的种类

        # 特征图大小
        total_num_anchors = pi.shape[2] * pi.shape[3]

        x_shifts = None
        y_shifts = None

        # exp_stride: 保存每一层的采样间隔，size=(1, grid.shape[1])
        exp_strides = torch.zeros(1, total_num_anchors).fill_(strides[layer_index]).type_as(pi[0])

        # 获取每个特征图的x,y格点坐标
        # x_shifts_per_img = x_shifts * exp_strides[layer_index]
        # y_shifts_per_img = y_shifts * exp_strides[layer_index]
        x_shifts_per_img = x_shifts[0] * exp_strides
        y_shifts_per_img = y_shifts[0] * exp_strides

        # 获取每个格点的中心坐标,左上角为原点, x_c_per_img.shape = [kinds_of_anchors, num_gt_per_batch, total_num_anchors]
        x_c_per_img = ((x_shifts_per_img + 0.5 * exp_strides).unsqueeze(0).repeat(kinds_of_anchors,num_gt, 1))
        y_c_per_img = ((y_shifts_per_img + 0.5 * exp_strides).unsqueeze(0).repeat(kinds_of_anchors,num_gt, 1))

        # 初步筛选
        # t.shape = (kinds_of_anchors, num_gt, [class,x,y,w,h,theta])
        gt_l = (t[:, :, :, 0] - 0.5 * t[:, :, :, 2])# shape=(kinds_of_anchor, num_gt_per_img, total_anchors)
        gt_t = (t[:, :, :, 1] - 0.5 * t[:, :, :, 3])
        gt_r = (t[:, :, :, 0] + 0.5 * t[:, :, :, 2])
        gt_b = (t[:, :, :, 1] + 0.5 * t[:, :, :, 3])
        # gt_b = ((t[:, :, :, 1] + 0.5 * t[:, :, :, 3]).unsqueeze(-1).repeat(1, 1, total_num_anchors))

        '''
        计算出每个anchor的左上角和右上角坐标（左上角为原点），然后与真实的标签值进行判断“当前anchor是否处于GT的内部”：如果是则为正样本
        '''
        bbox_l = x_c_per_img - gt_l
        bbox_r = gt_r - x_c_per_img
        bbox_t = y_c_per_img - gt_t
        bbox_b = gt_b - y_c_per_img
        bboxes = torch.stack([bbox_l, bbox_t, bbox_r, bbox_b], 3)  # size = (kinds_of_anchors, num_gt, feature_size1*feature_size2,4)

        # 然后将所有落在GT中的anchor挑选出来
        in_boxes = bboxes.min(dim=-1).values > 0.0  # 必须全部大于0才是需要的anchor
        in_boxes_all = in_boxes.sum(dim=0) > 0  # 中心点位于标注框内的锚框为True,相当于一个mask

        '''
        再次筛选：绘制一个边长为5的正方形。左上角点为（gt_l，gt_t），右下角点为（gt_r，gt_b）。gt以正方形范围形式去挑选锚框
        t.shape = [class,x,y,w,h,theta]
        '''
        radius = 1  # TODO：半径如果过大，似乎会过多增加样本，导致训练时间很长，尝试半径为1
        gt_l = (t[:, :, :, 0]) - radius * exp_strides # x - radius*stride
        gt_t = (t[:, :, :, 1]) - radius * exp_strides # y - radius*stride
        gt_r = (t[:, :, :, 0]) + radius * exp_strides
        gt_b = (t[:, :, :, 1]) + radius * exp_strides

        c_l = x_c_per_img - gt_l
        c_r = gt_r - x_c_per_img
        c_t = y_c_per_img - gt_t
        c_b = gt_b - y_c_per_img
        center = torch.stack([c_l, c_t, c_r, c_b], 3)
        in_centers = center.min(dim=-1).values > 0.0
        in_centers_all = in_centers.sum(dim=0) > 0

        # 某一边在gt里面
        # boxes_or_center.append(in_boxes_all | in_centers_all)
        boxes_or_center = (in_boxes_all & in_centers_all)  # 非 list版本

        # 两者都在gt里面
        # boxes_and_center.append(in_boxes[:, boxes_or_center[i]] & in_centers[:, boxes_or_center[i]])
        # boxes_and_center = (in_boxes[:, boxes_or_center] & in_centers[:, boxes_or_center]).unsqueeze(0).repeat(kinds_of_anchors,1,1)  # 非 list版本

        # return boxes_or_center, boxes_and_center
        boxes_or_center = boxes_or_center.repeat(kinds_of_anchors,1,1)
        return boxes_or_center

if __name__=='__main__':
    # Model
    device = torch.device("cuda")
    x = torch.randn(1,3,224,224).to(device)
    x = x.clone().detach()
    # Inference
