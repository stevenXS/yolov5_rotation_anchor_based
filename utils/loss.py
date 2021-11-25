# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""
import logging

import cv2
import torch
from utils.torch_utils import select_device
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import argparse
from utils.metrics import bbox_iou,bbox_iou_anchor_free
from utils.torch_utils import is_parallel
from utils.general import *
from models.yolo import Model

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# from general import *
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def gaussian_label(label, num_class, u=0, sig=4.0):
    '''
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    @param label: 当前box的θ类别  shape(1)
    @param num_class: θ类别数量=180
    @param u: 高斯函数中的μ
    @param sig: 高斯函数中的σ
    @return: 高斯离散数组:将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1 shape(180)
    '''
    # floor()返回数字的下舍整数   ceil() 函数返回数字的上入整数  range(-90,90)
    # 以num_class=180为例，生成从-90到89的数字整形list  shape(180)
    x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))  # shape(180) 为-90到89的经高斯公式计算后的数值
    # 将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1
    return np.concatenate([y_sig[math.ceil(num_class / 2) - int(label.item()):],
                           y_sig[:math.ceil(num_class / 2) - int(label.item())]], axis=0)



class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.class_index = 5 + model.nc

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma

        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEangle = FocalLoss(BCEangle, g)

        # 获取每一层的output
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEangle = BCEangle

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs=None, img_path=None):  # predictions, targets, model
        '''
        @param imgs: 额外新增的参数，可视化训练时期anchor和target的匹配过程
        '''
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        langle = torch.zeros(1, device=device)

        # build_targets函数返回的结果应该是正样本
        '''
        tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor tcls[i].shape=(num_i, 1)
            eg：tcls[0] = tensor([73, 73, 73])
        tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh tbox[i].shape=(num_i, 4)
            eg: tbox[0] = tensor([[ 0.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 1.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 0.19355,  1.27958,  4.38709, 14.92512]])
        indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
            1.(该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
            2.indices[i].shape=(4, num_i)
            eg： indices[0] = (tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([23, 23, 22]), tensor([2, 1, 2]))
        anch : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
            anchor[i].shape=(num_i, 2)
            eg：tensor([[2.00000, 3.75000],  [2.00000, 3.75000],  [2.00000, 3.75000]])
        tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的angle tensor
            tangle[i].shape=(num_i, 1)
            eg：tangle[0] = tensor([179, 179, 179])
        '''
        # tcls, tangle, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tangle, tbox, indices, anchors,txywh = self.build_targets(p, targets)  # 增加了匹配需要的txywh,add

        # add------------------------------------
        if imgs is not None:
            # vis_bbox(imgs,targets)  #可视化anchor的匹配过程，add
            vis_match(imgs,targets,tcls,tbox,indices,anchors,p,txywh,img_path) # 可视化anchor的匹配关系，2021-10-18 14:31:31

        # Losses
        # pi.size= (batch_size,channel, feature_map_size1,feature_map_size2, [x,y,w,h,obj]+ num_classes +180angle)
        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            b: 当前batch中的图片索引
            a: 每个anchor的索引
            gj: 每个anchor的y坐标
            gi: 每个anchor的x坐标
            '''
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets

            if n:
                # ps.size=(num_anchors, classes
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box

                # 使用到了CIou loss
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)

                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:self.class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp # 被匹配到的类别就置为1
                    lcls += self.BCEcls(ps[:, 5:self.class_index], t)  # BCE
                
                # Θ类别损失
                ttheta = torch.zeros_like(ps[:, self.class_index:])  # size(num, 180)

                for idx in range(len(ps)):  # idx start from 0 to len(ps)-1
                    # 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor  tangle[i].shape=(num_i, 1)
                    theta = tangle[i][idx]  # 取出第i个layer中的第idx个目标的角度数值  例如取值θ=90
                    # CSL论文中窗口半径为6效果最佳，过小无法学到角度信息，过大则角度预测偏差加大
                    csl_label = gaussian_label(theta, 180, u=0, sig=6)  # 用长度为1的θ值构建长度为180的csl_label
                    ttheta[idx] = torch.from_numpy(csl_label)  # 将cls_label放入对应的目标中

                langle += self.BCEangle(ps[:, self.class_index:], ttheta)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        langle *= self.hyp['angle']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + langle) * bs, torch.cat((lbox, lobj, lcls, langle)).detach()
    '''
        预测的anchor与真实标签做对比，进行筛选
    '''
    def build_targets(self, p, targets):
        '''
        @param p: list: [small_forward, medium_forward, large_forward]
            eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
        @param targets: size=(image_id,class,x,y,w,h,theta), image_id表示当前target属于batch的哪一张
        @param targets: torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, Θ])
        @param model: 模型
            Returns:
                tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor
                               tcls[i].shape=(num_i, 1)
                           eg：tcls[0] = tensor([73, 73, 73])
                tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh
                               tbox[i].shape=(num_i, 4)
                           eg: tbox[0] = tensor([[ 0.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 1.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 0.19355,  1.27958,  4.38709, 14.92512]])
                indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
                               (该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
                                     indices[i].shape=(4, num_i)
                                eg： indices[0] = (tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([23, 23, 22]), tensor([2, 1, 2]))
                anch : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
                                    anchor[i].shape=(num_i, 2)
                                eg：tensor([[2.00000, 3.75000],  [2.00000, 3.75000],  [2.00000, 3.75000]])
                tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的angle tensor
                               tangle[i].shape=(num_i, 1)
                           eg：tangle[0] = tensor([179, 179, 179])
        '''

        na, nt = self.na, targets.shape[0]  # 预测框的种类, 标签值的数量
        tcls, tbox, indices, anch = [], [], [], []
        tangle = []
        txywh = [] # add, 为可视化提供xy,wh

        gain = torch.ones(8, device=targets.device)  # normalized to grid space gain

        '''
        na是anchor的数量，假如na=3，那么torch.arange(na) = tensor[0,1,2];
        然后转化成float型数据，最后维度展开为(3,1)=tensor[[0.],[1.],[2.]];
        repeat(1,nt),沿着上面的第二个维度进行复制nt次，nt=真实标签的数量;
        所以ai表示anchor的索引ai = anchor_index = (na，nt).
        '''
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)

        '''
        1.ai[:, :, None]:这里将原本2个维度的ai增加了一个维度=(na,nt,1);
        2.targets.repeat(na, 1, 1): target一共两个维度(number_of_targets，[image,class,x,y,w,h])进行维度扩展，假如na=3，那么扩展后的维度
            targets.repeat(na, 1, 1)=(na, nt, [image,class,x,y,w,h])
        3.再将两个三维矩阵在第2个维度进行进行拼接，即(na, nt, 1+6)
        eg: 目的是为了将预测框和标签值进行矩阵拼接，接下来进行筛选策略。
        4.targets此时的size = (num_anchor, num_gt, [img_id, cls_id, x,y,w,h,theta, anchor_index])
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices


        '''
        off偏置矩阵，获取上下左右四个点及当前的点
            tensor([[ 0.00000,  0.00000],
            [ 0.50000,  0.00000],
            [ 0.00000,  0.50000],
            [-0.50000,  0.00000],
            [ 0.00000, -0.50000]])
        '''
        g = 0.5  # bias
        off = torch.tensor([[0, 0],[1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 对三个特征层依次处理，anchor也进行同比例缩放，8,16,32
        for i in range(self.nl):
            # 获取当前特征层的预设的anchor，anchor based版本有三种预设值，所以anchors是一个list
            # anchors size=（3中scale, 2(w,h)）
            anchors = self.anchors[i]

            # gain[2:6]全是80，
            # torch.tensor(p[i].shape)[[3, 2, 3, 2]]获取p[i]层的对应索引位的维度数，第3个维度，第2个维度···
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # targets已经被归一化处理，然后通过gain矩阵中获取的特征图大小进行映射，投影到特征图上去
            t = targets * gain

            if nt:  # 标签数量
                # 匹配策略
                '''
                # t[:, :, 4:6]: 索引标签值的第4，5维数据，即w,h；
                # anchors[:, None]: 扩张第二个维度，（3,2）->(3,1,2)
                # r: 获得宽高比
                '''
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                # 如果每个标签值和anchor的wh比最大值小于超参数里面的预设值，则该anchor为合适的anchor
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 进行过滤

                # Offsets
                # 以图像左上角为原点，gxy为标签值的x,y坐标
                # 然后转化为以特征图右下角为原点，即target[x,y] -> (80-x, 80-y)
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse

                # gxi：转换后的标签的xy坐标（右下角为原点），gxy: 真实标签的xy左边（左上角为原点）
                # j,l矩阵互斥，k,m矩阵互斥
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T # 判断转换前的x，y是否大于1，并且x距它左边,y距它上边的网格距离是否<0.5?如果都满足条件，则选中
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T # 同理对转换后的坐标判断x距它网格的右边，y距它网格下边是否同时满足上述两个条件。

                # 然后j是一个bool变量的矩阵，size=（5，标签的数量）,假设(5,15)
                # 然后将这几个矩阵进行连接起来
                j = torch.stack((torch.ones_like(j), j, k, l, m))

                # t.repeat((5, 1, 1))：在（15,8）第0维前重复5次->(5,15,8)
                # 这里对正样本进行了扩充
                t = t.repeat((5, 1, 1))[j] # 过滤后，t剩下两个维度，[select_anchors, 8], 即筛选出来的anchor

                # 获取所有标签值的偏置
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # 获取每个标签的图像，类别索引，第0,1维度
            angle = t[:, 6].long()  # 获取角度索引，第6个维度
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long() # 获取被匹配的anchor的x,y坐标
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 7].long()  # 每个anchor的索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 保存图片序号、anchor索引、网格点坐标
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 获取xy相对于网格点的偏置，以及box宽高
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tangle.append(angle)  # angle
            txywh.append(torch.cat((gxy,gwh),1)) # 可视化增加xy,wh
        return tcls, tangle, tbox, indices, anch , txywh


class ComputeLoss_AnchorFree:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.class_index = 5 + model.nc

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEangle = FocalLoss(BCEangle, g)

        # 获取每一层的output
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEangle = BCEangle

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        langle = torch.zeros(1, device=device)

        # build_targets函数返回的结果应该是正样本
        '''
        tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor tcls[i].shape=(num_i, 1)
            eg：tcls[0] = tensor([73, 73, 73])
        tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh tbox[i].shape=(num_i, 4)
            eg: tbox[0] = tensor([[ 0.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 1.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 0.19355,  1.27958,  4.38709, 14.92512]])
        indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
            1.(该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
            2.indices[i].shape=(4, num_i)
            eg： indices[0] = (tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([23, 23, 22]), tensor([2, 1, 2]))
        anch : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
            anchor[i].shape=(num_i, 2)
            eg：tensor([[2.00000, 3.75000],  [2.00000, 3.75000],  [2.00000, 3.75000]])
        tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的angle tensor
            tangle[i].shape=(num_i, 1)
            eg：tangle[0] = tensor([179, 179, 179])
        '''
        tcls, tangle, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # 可视化target，2021-10-18 14:28:36
        # vis_bbox(p,targets)

        # 可视化anchor的匹配关系，2021-10-18 14:31:31
        # vis_match(p,targets,tcls,tbox,indices,anchors)

        # Losses
        # pi.size= (batch_size,channel, feature_map_size1,feature_map_size2, [x,y,w,h,obj]+ num_classes +180angle)
        for i, pi in enumerate(p):  # layer index, layer predictions
            '''
            b: 当前batch中的图片索引
            a: 每个anchor的索引
            gj: 每个anchor的y坐标
            gi: 每个anchor的x坐标
            '''
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # ps.size=(num_anchors, classes
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box

                # 使用到了CIou loss
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)

                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:self.class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:self.class_index], t)  # BCE

                # Θ类别损失
                ttheta = torch.zeros_like(ps[:, self.class_index:])  # size(num, 180)

                for idx in range(len(ps)):  # idx start from 0 to len(ps)-1
                    # 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor  tangle[i].shape=(num_i, 1)
                    theta = tangle[i][idx]  # 取出第i个layer中的第idx个目标的角度数值  例如取值θ=90
                    # CSL论文中窗口半径为6效果最佳，过小无法学到角度信息，过大则角度预测偏差加大
                    csl_label = gaussian_label(theta, 180, u=0, sig=6)  # 用长度为1的θ值构建长度为180的csl_label
                    ttheta[idx] = torch.from_numpy(csl_label)  # 将cls_label放入对应的目标中

                langle += self.BCEangle(ps[:, self.class_index:], ttheta)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        langle *= self.hyp['angle']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + langle) * bs, torch.cat((lbox, lobj, lcls, langle)).detach()

    '''
        预测的anchor与真实标签做对比，进行筛选
    '''

    def build_targets(self, p, targets):
        '''
        @param p: list: [small_forward, medium_forward, large_forward]
            eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
        @param targets: size=(image,class,x,y,w,h,theta)
        @param targets: torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, Θ])
        @param model: 模型
            Returns:
                tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor
                               tcls[i].shape=(num_i, 1)
                           eg：tcls[0] = tensor([73, 73, 73])
                tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh
                               tbox[i].shape=(num_i, 4)
                           eg: tbox[0] = tensor([[ 0.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 1.19355,  0.27958,  4.38709, 14.92512],
                                                 [ 0.19355,  1.27958,  4.38709, 14.92512]])
                indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
                               (该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
                                     indices[i].shape=(4, num_i)
                                eg： indices[0] = (tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([23, 23, 22]), tensor([2, 1, 2]))
                anch : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
                                    anchor[i].shape=(num_i, 2)
                                eg：tensor([[2.00000, 3.75000],  [2.00000, 3.75000],  [2.00000, 3.75000]])
                tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的angle tensor
                               tangle[i].shape=(num_i, 1)
                           eg：tangle[0] = tensor([179, 179, 179])
        '''

        na, nt = self.na, targets.shape[0]  # 预测框的种类, 标签值的数量
        tcls, tbox, indices, anch = [], [], [], []
        tangle = []
        gain = torch.ones(8, device=targets.device)  # normalized to grid space gain

        '''
        na是anchor的数量，假如na=3，那么torch.arange(na) = tensor[0,1,2];
        然后转化成float型数据，最后维度展开为(3,1)=tensor[[0.],[1.],[2.]];
        repeat(1,nt),沿着上面的第二个维度进行复制nt次，nt=真实标签的数量;
        所以ai表示anchor的索引ai = anchor_index = (na，nt).
        '''
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)

        '''
        1.ai[:, :, None]:这里将原本2个维度的ai增加了一个维度=(na,nt,1);
        2.targets.repeat(na, 1, 1): target一共两个维度(number_of_targets，[image,class,x,y,w,h])进行维度扩展，假如na=3，那么扩展后的维度
            targets.repeat(na, 1, 1)=(na, nt, [image,class,x,y,w,h])
        3.再将两个三维矩阵在第2个维度进行进行拼接，即(na, nt, 1+6)
        eg: 目的是为了将预测框和标签值进行矩阵拼接，接下来进行筛选策略。
        4.targets此时的size = (num_anchor, num_gt, [img_id, cls_id, x,y,w,h,theta, anchor_index])
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        '''
        off偏置矩阵，获取上下左右四个点及当前的点
            tensor([[ 0.00000,  0.00000],
            [ 0.50000,  0.00000],
            [ 0.00000,  0.50000],
            [-0.50000,  0.00000],
            [ 0.00000, -0.50000]])
        '''
        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 对三个特征层依次处理，anchor也进行同比例缩放，8,16,32
        for i in range(self.nl):
            # 获取当前特征层的预设的anchor，anchor based版本有三种预设值，所以anchors是一个list
            # anchors size=（3中scale, 2(w,h)）
            anchors = self.anchors[i]

            # gain[2:6]全是80，
            # torch.tensor(p[i].shape)[[3, 2, 3, 2]]获取p[i]层的对应索引位的维度数，第3个维度，第2个维度···
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # targets已经被归一化处理，然后通过gain矩阵中获取的特征图大小进行映射，投影到特征图上去
            t = targets * gain

            if nt:  # 标签数量
                # 匹配策略
                '''
                # t[:, :, 4:6]: 索引标签值的第4，5维数据，即w,h；
                # anchors[:, None]: 扩张第二个维度，（3,2）->(3,1,2)
                # r: 获得宽高比
                '''
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                # 如果每个标签值和anchor的wh比最大值小于超参数里面的预设值，则该anchor为合适的anchor
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 进行过滤

                # Offsets
                # 以图像左上角为原点，gxy为标签值的x,y坐标
                # 然后转化为以特征图右下角为原点，即target[x,y] -> (80-x, 80-y)
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse

                # gxi：转换后的标签的xy坐标（右下角为原点），gxy: 真实标签的xy左边（左上角为原点）
                # j,l矩阵互斥，k,m矩阵互斥
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 判断转换前的x，y是否大于1，并且x距它左边,y距它上边的网格距离是否<0.5?如果都满足条件，则选中
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 同理对转换后的坐标判断x距它网格的右边，y距它网格下边是否同时满足上述两个条件。

                # 然后j是一个bool变量的矩阵，size=（5，标签的数量）,假设(5,15)
                # 然后将这几个矩阵进行连接起来
                j = torch.stack((torch.ones_like(j), j, k, l, m))

                # t.repeat((5, 1, 1))：在（15,8）第0维前重复5次->(5,15,8)
                # 这里获取已经匹配到了的anchor
                t = t.repeat((5, 1, 1))[j]  # 过滤后，t剩下两个维度，[select_anchors, 8], 即筛选出来的anchor

                # 获取所有标签值的偏置
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # 获取每个标签的图像，类别索引，第0,1维度
            angle = t[:, 6].long()  # 获取角度索引，第6个维度
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 获取每个box所在网格点的坐标
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 7].long()  # 每个anchor的索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 保存图片序号、anchor索引、网格点坐标
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 获取xy相对于网格点的偏置，以及box宽高
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tangle.append(angle)  # angle

        return tcls, tangle, tbox, indices, anch

'''
2021年11月10日14:16:50
功能：解耦合后的损失函数
'''

class KLDloss(nn.Module):
    def __init__(self, taf=1.0, reduction="none"):
        super(KLDloss, self).__init__()
        self.reduction = reduction
        self.taf = taf

    def forward(self, pred, target):
        # pred [[x,y,w,h,angle], ...]
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        pre_angle_radian = 3.141592653589793 * pred[:, 4] / 180.0
        targrt_angle_radian = 3.141592653589793 * target[:, 4] / 180.0
        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld =  0.5 * (
                        4 * torch.pow( ( delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
                      + 4 * torch.pow( ( delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
                     )\
             + 0.5 * (
                        torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                     )\
             + 0.5 * (
                        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                     )\
             - 1.0

        kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))
        # return kld_loss
        return kld_loss.sigmoid()

class ComputeLoss_AnchorFree_Decoupled:
        # Compute losses
        def __init__(self, model, autobalance=False):
            self.sort_obj_iou = False
            device = next(model.parameters()).device  # get model device
            h = model.hyp  # hyperparameters
            self.class_index = 5 + model.nc
            self.model = model # 复制model，方便后面索引，add

            # Define criteria
            # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
            # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
            # BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)
            BCEcls = nn.BCEWithLogitsLoss(reduction="none")
            BCEobj = nn.BCEWithLogitsLoss(reduction="none")
            BCEangle = nn.BCEWithLogitsLoss(reduction="none").to(device)

            # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
            self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

            # Focal loss
            g = h['fl_gamma']  # focal loss gamma
            if g > 0:
                BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
                BCEangle = FocalLoss(BCEangle, g)

            # 获取每一层的output
            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
            self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
            self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
            self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
            self.BCEangle = BCEangle
            self.iou_loss = KLDloss() # add

            # 设置属性
            for k in 'na', 'nc', 'nl', 'anchors':
                setattr(self, k, getattr(det, k))

        def __call__(self, p, targets, imgs=None, img_path=None):  # predictions, targets, model, 增加了两个可视化参数imgs, img_path
            '''
                @param p: list: [small_forward, medium_forward, large_forward]
                    eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
                @param targets: shape=(num_gt, [batch_size, class_id, x,y,w,h,theta])
            '''
            device = targets.device
            lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1,device=device)


            # 获取三层feature map分别对应的匹配框的属性
            cls_targets, reg_targets, angle_targets, obj_targets, anchor_mask = self.build_targets_anchor_free(p, targets)  # targets


            # 将输出的list进行拼接，每个list包含了batch_size个输出
            cls_targets = torch.cat(cls_targets, 0)
            reg_targets = torch.cat(reg_targets, 0)
            angle_targets = torch.cat(angle_targets, 0) # add (n_anchor_select_final)
            obj_targets = torch.cat(obj_targets, 0)
            anchor_masks = torch.cat(anchor_mask,0)

            # 将output也全部torch.cat
            all_outputs = torch.cat(p, 1)
            bbox_preds_with_angle = all_outputs[:, :, :5]  # [batch, anchors_all, 5]
            obj_preds = all_outputs[:, :, 5].unsqueeze(-1)  # [batch, nanchors_all, 1]
            cls_preds = all_outputs[:, :, 6:]  # [batch, anchors_all, n_cls]

            reg_targets_with_angle = torch.cat((reg_targets, angle_targets.unsqueeze(-1)), dim=1)

            '''
            eg: loss这里不能除以num_gt，因为num_gt可能为0
            '''
            lbox += (self.iou_loss(reg_targets_with_angle, bbox_preds_with_angle.view(-1, 5)[anchor_masks])).sum() # reg_targets_with_angle.shape = (18,5), bbox_preds_with_angle.view(-1, 5).shape = (400,5)
            lcls += (self.BCEcls(cls_preds.view(-1, self.nc)[anchor_masks], cls_targets)).sum()
            lobj += (self.BCEobj(obj_preds.view(-1, 1)[anchor_masks], obj_targets[anchor_masks].float())).sum()  # obj_preds[i].shape=(total_anchors,1) obj_targets[i].shape=(total_anchors,1)
            # lobj += (self.BCEobj(obj_preds.view(-1, 1)[anchor_masks], obj_targets[anchor_masks].float())).sum()  # obj_preds[i].shape=(total_anchors,1) obj_targets[i].shape=(total_anchors,1)

            lbox *= self.hyp['box']
            lobj *= self.hyp['obj']
            lcls *= self.hyp['cls']
            reg_weight = 5.0
            # print("lbox:",lbox)
            # return (reg_weight*lbox + lobj + lcls), torch.cat((reg_weight*lbox, lobj, lcls)).detach()
            return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach() # 因为 loss不参与更新，所以直接detach()

        # 就类似于get_assignments()方法-yolox
        def build_targets_anchor_free(self, p, targets):
            '''
            @param p: list: [small_forward, medium_forward, large_forward]
                eg: small_forward.size=( batch_size, 1种scale框*size1*size2, [x,y,w,h,theta,obj,classes])
            @param targets: size=(num_gt, [image_id,class,x,y,w,h,theta]), image_id表示当前target属于batch的哪一张
                eg: torch.Size = (num_gt:当前batch中所有目标数量, [该image属于该batch的第几个图片, classId, xywh, Θ])
            '''

            cls_targets = []
            reg_targets = []
            angle_targets = []
            obj_targets = []
            anchor_masks = []

            # 获取预测的输出
            for layer_id, pi in enumerate(p):
                # 得到每一个输出层的属性，一共三层
                # bboxes_preds = pi[:, :, 0:4] # shape = (batch, size1*size2, [x,y,w,h,theta,obj,class])
                # angle_preds = pi[:, :, 4]
                # obj_preds = pi[:, :, 5]
                # cls_preds = pi[:, :, 6:]

                # 逐个batch的处理
                for batch_id in range(pi.shape[0]):
                    # TODO:eg：由于targets的第二个维度中的第一列是batch的id，所以需要加一个掩膜，来索引对应batch的GT
                    gt_batch_mask = (targets[:, 0] == batch_id) # gt_batch_mask.shape = (当前batch的num_gt, 7)
                    batch_targets = targets[gt_batch_mask] # 获取某一个batch的GT值
                    num_gt_per_batch = batch_targets.shape[0]  # 当前image中GT的数量,shape = (num_gt_per_batch,[class,x,y,w,h,theta])
                    batch_pi = pi[batch_id] # 当前batch中的输出，shape=(1种scale框*size1*size2, [x,y,w,h,theta,obj,classes])

                    # 获取某一个batch的输出
                    bboxes_preds = batch_pi[:, 0:4]  # shape = (batch, size1*size2, [x,y,w,h,theta,obj,class])
                    angle_preds = batch_pi[:, 4]
                    obj_preds = batch_pi[:, 5]
                    cls_preds = batch_pi[:, 6:]

                    if num_gt_per_batch:
                        # TODO：初筛操作,anchor_mask.shape=(size1*size2),in_boxes_and_center.shape=(num_gt_per_batch, size1*size2)
                        anchor_mask, in_boxes_and_center = self.get_anchor_info(batch_pi, batch_targets, num_gt_per_batch, layer_index=layer_id)

                        # 获取真实值的信息
                        gt_bboxes = batch_targets[:num_gt_per_batch, 2:6]
                        gt_angles = batch_targets[:num_gt_per_batch, 6]
                        gt_classes = batch_targets[:num_gt_per_batch, 1] #

                        # TODO：根据初筛得到的mask将网络的输出进行“初步筛选”
                        bboxes_preds = bboxes_preds[anchor_mask] # shape = (num_select, 4)
                        angle_preds = angle_preds[anchor_mask] # shape = (num_select,1)
                        obj_preds = obj_preds[anchor_mask]
                        cls_preds = cls_preds[anchor_mask]
                        num_in_anchor = bboxes_preds.shape[0]

                        # 预测的bbox+angle，真实值的bbox+angle，进行concat为后续作loss做准备
                        gt_bboxes_with_angle = torch.cat((gt_bboxes, gt_angles.unsqueeze(1)), dim=1) # shape=（num_gt，4+1）
                        pred_bboxes_with_angle = torch.cat((bboxes_preds,angle_preds.unsqueeze(1)), dim=1)# shape=（num_select，4+1）

                        # TODO: 将初步筛选的bbox（num_select,5)与真实值（num_gt,5）作loss
                        # pairwise_iou_loss.shape = (num_in_anchor, num_select)
                        pairwise_iou_loss = self.compute_kld_loss(pred_bboxes_with_angle, gt_bboxes_with_angle)
                        pairwise_iou_approximate = 1.0 - pairwise_iou_loss # 取loss的近似值

                        # 将使用F.one_hot将标签的维度转化为与预测值的维度一样的矩阵
                        gt_cls_per_image = (
                            F.one_hot(gt_classes.to(torch.int64), self.nc) # shape=(num_gt, 16)
                            .float()
                            .unsqueeze(1) # shape = (num_gt,1,16)
                            .repeat(1, num_in_anchor, 1) # shape = (num_gt,num_in_anchor,16)
                        )
                        
                        # 处理预测的cls, cls的条件概率和obj的先验概率做乘积，得到目标的类别分数。
                        cls_preds = (
                            cls_preds.float().unsqueeze(0).repeat(num_gt_per_batch, 1, 1).sigmoid_()
                            * obj_preds.unsqueeze(1).unsqueeze(0).repeat(num_gt_per_batch, 1, 1).sigmoid_()
                        ) # cls_preds shape: (n_gt, n_anchor_select, 16)

                        # 将所有的标签值cls与所有候选框的cls进行做交叉熵损失
                        pairwise_cls_loss = F.binary_cross_entropy_with_logits(cls_preds.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)  # cls_preds_ shape: (n_gt, n_anchor_select)
                        del cls_preds

                        # 让中心不在标注框里或者中心不在5*5方格里的锚框cost值很大
                        # 这里做成本计算，即【分类损失】和【回归损失】
                        cost = (pairwise_cls_loss + 3.0 * pairwise_iou_loss+ 100000.0 * (~in_boxes_and_center))
            
                        # 这里引入SIMOTA策略（旷世）
                        #     1.首先设置候选框的数量，这里会新建一个矩阵，shape=(num_gt, num_select)；
                        #     2.通过cost成本挑选候选框，然后通过topk_ious筛选的信息，动态的选择候选框；
                        # return :
                        #     返回匹配到的预测框数量，匹配到的cls, 匹配到的iou,匹配到的gt的index
                        #     num_matched_anchors, matched_classes_per_gt,matched_ious_per_gt, matched_gt_index

                        # TODO: self.dynamic_k_matching()该方法目前有bug
                        (num_matched_anchors, matched_classes_per_gt, matched_ious_per_gt, matched_gt_index, mask_in_boxes)=self.dynamic_k_matching(cost, pairwise_iou_approximate, gt_classes, num_gt_per_batch, anchor_mask)


                        cls_target = F.one_hot(matched_classes_per_gt.to(torch.int64), self.nc) * matched_ious_per_gt.unsqueeze(-1)# (num_select_final, self.nc)
                        obj_target = anchor_mask.unsqueeze(-1)  # （n_anchor_select_final, 1）
                        # obj_target = mask_in_boxes.unsqueeze(-1)  # （n_anchor_select_final, 1）
                        reg_target = gt_bboxes[matched_gt_index]  # (n_anchor_select_final, 4)
                        angle_target = gt_angles[matched_gt_index]  # add  （n_anchor_select_final）

                        # 直接用初步筛选得到的正样本进行回归和分类
                    else:
                        cls_target = pi.new_zeros((0, self.nc))
                        reg_target = pi.new_zeros((0, 4))
                        angle_target = pi.new_zeros(0)
                        obj_target = pi.new_zeros((pi.shape[1], 1))
                        anchor_mask = pi.new_zeros(pi.shape[1]).bool()
                    cls_targets.append(cls_target)
                    reg_targets.append(reg_target)
                    angle_targets.append(angle_target)  # add
                    obj_targets.append(obj_target)
                    anchor_masks.append(anchor_mask)

            return cls_targets, reg_targets, angle_targets, obj_targets, anchor_masks

        '''
        功能：初步筛选正样本的anchor，为精细化筛选做准备
        '''
        def get_anchor_info(self, pi, targets, num_gt_per_batch, layer_index):
            '''
            @param pi: 传入的是batch中某一个的特征图，size = (1种scale框 * size1 * size2, [x,y,w,h,theta,obj,classes])
                eg: 将检测头的每一层输出的特征图w，h进行合并
            @param targets: 当前batch中某一个image的targets，shape=(num_gt_per_batch, [class,x,y,w,h,theta])
            '''
            exp_strides = []  # 保存采样率
            strides = [8, 16, 32]  # 三层layer的采样率
            num_anchor, num_gt_per_batch = self.na, num_gt_per_batch  # 预测框的种类, 当前image中标签值的数量

            # 返回所有符合条件的正样本anchor，它是一个list，size=3
            boxes_or_center = []
            boxes_and_center = []

            # 初始化网格空间的缩放增益
            gain = torch.ones(7, device=targets.device)
            xx = [1,int(np.sqrt(pi.shape[0])), int(np.sqrt(pi.shape[0])), 1] # (w,h,22)
            gain[2:6] = torch.tensor(xx)[[2, 1, 2, 1]]  # 特征图的w,h,w,h

            # targets: size=(num_anchor, num_gt,[image,class_id,x,y,w,h,theta])
            t = targets * gain  # 将标签值缩放到特征图上

            # 所有的anchor数量
            total_num_anchors = pi.shape[0]

            # 获取grid的x,y坐标
            if RANK !=-1: # DDP
                x_shifts = self.model.module.model[-1].grid[layer_index][:, :, 0]
                y_shifts = self.model.module.model[-1].grid[layer_index][:, :, 1]
            else:
                x_shifts = self.model.model[-1].grid[layer_index][:, :, 0] # shape = (1, size1*size2)
                y_shifts = self.model.model[-1].grid[layer_index][:, :, 1]

            # exp_stride: 保存每一层的采样间隔，size=(1, grid.shape[1])
            exp_strides = torch.zeros(1, pi.shape[0]).fill_(strides[layer_index]).type_as(pi[0])

            # 获取每个特征图的x,y格点坐标
            # x_shifts_per_img = x_shifts * exp_strides[layer_index]
            # y_shifts_per_img = y_shifts * exp_strides[layer_index]
            x_shifts_per_img = x_shifts[0]  * exp_strides
            y_shifts_per_img = y_shifts[0]  * exp_strides

            # 获取每个格点的中心坐标,左上角为原点
            # x_c_per_img.shape = [num_gt_per_batch, total_num_anchors]
            x_c_per_img = ((x_shifts_per_img + 0.5 * exp_strides).repeat(num_gt_per_batch, 1))
            y_c_per_img = ((y_shifts_per_img + 0.5 * exp_strides).repeat(num_gt_per_batch, 1))

            # TODO：初步筛选
            # 计算标签的左上角和右下角坐标，即（[left,top], [right,bottom])
            # gt_xxx与x_c_per_img维度应该相同
            # gt_l = ( (t[: , :, 2] - 0.5 * t[:, :, 4]).squeeze(1).repeat(1,total_num_anchors))
            # gt_t = ( (t[: , :, 3] - 0.5 * t[:, :, 5]).squeeze(1).repeat(1,total_num_anchors))
            # gt_r = ( (t[: , :, 2] + 0.5 * t[:, :, 4]).squeeze(1).repeat(1,total_num_anchors))
            # gt_b = ( (t[: , :, 3] + 0.5 * t[:, :, 5]).squeeze(1).repeat(1,total_num_anchors))
            gt_l = ((t[:, 2] - 0.5 * t[:, 4]).unsqueeze(1).repeat(1, total_num_anchors)) # shape=(num_gt_per_img, total_anchors)
            gt_t = ((t[:, 3] - 0.5 * t[:, 5]).unsqueeze(1).repeat(1, total_num_anchors))
            gt_r = ((t[:, 2] + 0.5 * t[:, 4]).unsqueeze(1).repeat(1, total_num_anchors))
            gt_b = ((t[:, 3] + 0.5 * t[:, 5]).unsqueeze(1).repeat(1, total_num_anchors))

            # TODO ：anchor free的思路是每个格点都作为一个预测的anchor，因此anchor数量就是当前的特征图大小;
            # 计算出每个anchor的左上角和右上角坐标（左上角为原点），然后与真实的标签值进行判断“当前anchor是否处于GT的内部”：如果是则为正样本
            bbox_l = x_c_per_img - gt_l
            bbox_r = gt_r - x_c_per_img
            bbox_t = y_c_per_img - gt_t
            bbox_b = gt_b - y_c_per_img
            bboxes = torch.stack([bbox_l, bbox_t, bbox_r, bbox_b], 2)  # size = (num_gt, feature_size1*feature_size2,4)

            # 然后将所有落在GT中的anchor挑选出来
            in_boxes = bboxes.min(dim=-1).values > 0.0  # 必须全部大于0才是需要的anchor
            in_boxes_all = in_boxes.sum(dim=0) > 0  # 中心点位于标注框内的锚框为True,相当于一个mask

            # TODO 再次筛选：绘制一个边长为5的正方形。左上角点为（gt_l，gt_t），右下角点为（gt_r，gt_b）。gt以正方形范围形式去挑选锚框
            radius = 2  # 半径
            # gt_l = (targets[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides[i]# x - radius*stride
            # gt_r = (targets[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[i]
            # gt_t = (targets[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides[i]
            # gt_b = (targets[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[i]
            gt_l = (t[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides  # x - radius*stride
            gt_t = (t[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides
            gt_r = (t[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides
            gt_b = (t[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides
            # gt_b = (t[:, :, 3].squeeze(0).unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[layer_index]

            c_l = x_c_per_img - gt_l
            c_r = gt_r - x_c_per_img
            c_t = y_c_per_img - gt_t
            c_b = gt_b - y_c_per_img
            center = torch.stack([c_l, c_t, c_r, c_b], 2)
            in_centers = center.min(dim=-1).values > 0.0
            in_centers_all = in_centers.sum(dim=0) > 0

            # 某一边在gt里面
            # boxes_or_center.append(in_boxes_all | in_centers_all)
            boxes_or_center = in_boxes_all | in_centers_all #非 list版本

            # 两者都在gt里面
            # boxes_and_center.append(in_boxes[:, boxes_or_center[i]] & in_centers[:, boxes_or_center[i]])
            boxes_and_center= in_boxes[:, boxes_or_center] & in_centers[:, boxes_or_center] #非 list版本

            return boxes_or_center, boxes_and_center


        def compute_kld_loss(self,p, targets, taf = 1.0):
            '''
            @param p: 通过get_anchor_info（）的mask初筛得到的正样本，shape=(num_select, xywh+angle)
            @param targets: 标签值, shape = (num_gt, xywh,angle)
            @return
            '''
            with torch.no_grad():
                # 初始化标签值，一个zero矩阵，shape=(0,num_select)
                kld_loss_ = torch.zeros(0, p.shape[0], device=targets.device)
                for t in targets: # 将每一个真实值与初筛得到的正样本进行损失计算
                    t = t.unsqueeze(0).repeat(p.shape[0],1) # t.shape = (num_select, num_gt, 5)
                    kld_loss = self.kld_loss(p, t)
                    kld_loss_ = torch.cat((kld_loss_, kld_loss.unsqueeze(0)), dim=0)

            return kld_loss_

        # 这里损失的计算可以替换，add
        def kld_loss(self, p, t, taf = 1.0):
            '''
            @param p: 通过get_anchor_info（）的mask初筛得到的正样本，shape=(num_select,5)
            @param targets: 标签值, shape = (num_select, num_gt, 5)
            '''
            assert p.shape[0] == t.shape[0] # 断言操作
            p = p.view(-1,5)
            t = t.view(-1,5)

            delta_x = p[:, 0] - t[:, 0]
            delta_y = p[:, 1] - t[:, 1]

            # 角度转弧度，eg: 标签和预测的angle都是角度，需要进行转化，然后作坐标的三角函数变换
            p_angle_radian = 3.1415926535897932 * p[:, 4] / 180.0
            t_angle_radian = 3.1415926535897932 * t[:, 4] / 180.0
            delta_angle_radian = p_angle_radian - t_angle_radian

            # 进行带角度的损失函数计算
            kld = \
            0.5 * (
                  4 * torch.pow((delta_x.mul(torch.cos(t_angle_radian)) + delta_y.mul(torch.sin(t_angle_radian))),2) / torch.pow(t[:, 2], 2)
                + 4 * torch.pow((delta_y.mul(torch.cos(t_angle_radian)) - delta_x.mul(torch.sin(t_angle_radian))),2) / torch.pow(t[:, 3], 2)
            ) \
            + 0.5 * (
                    torch.pow(p[:, 3], 2) / torch.pow(t[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(p[:, 2], 2) / torch.pow(t[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(p[:, 3], 2) / torch.pow(t[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                  + torch.pow(p[:, 2], 2) / torch.pow(t[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
            ) \
            + 0.5 * (
                      torch.log(torch.pow(t[:, 3], 2) / torch.pow(p[:, 3], 2))
                    + torch.log(torch.pow(t[:, 2], 2) / torch.pow(p[:, 2], 2))
              ) - 1.0

            kld_loss = 1 - 1 / (taf + torch.log(kld + 1))
            return kld_loss
            # return kld_loss.sigmoid()

        # 旷世提出的SimOTA策略，会引入额外复杂度
        def dynamic_k_matching(self, cost, pairwise_iou_approximate, gt_classes, num_gt, anchor_mask):
            # 创建一个矩阵，shape=(num_gt, num_select)
            matching_matrix = torch.zeros_like(cost)

            iou_in_boxes = pairwise_iou_approximate # shape(num_gt, num_select)
            # TODO：1.设置前K个候选框,源代码k=10，这里考虑到速度问题取5
            top_k_anchor = min(10, iou_in_boxes.size(1))

            # 然后给每个目标挑选前K个候选框，topk_anchor_per_gt.shape = (num_gt, k)
            # topk_anchor_per_gt里面存储了每个GT对应的K个框的概率
            topk_anchor_per_gt, _ = torch.topk(iou_in_boxes, top_k_anchor, dim=1)

            '''
                eg: torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) :
                    功能：沿着给定的维度，返回输入的张量中前K个最大值，如果不指定维度，则默认返回最后一个维度；
                    k: 返回的前K个；
                    large：True-返回前K个最大值，False-返回前K个最小值
                    return：一个元组
            '''
            dynamic_k = torch.clamp(topk_anchor_per_gt.sum(1).int(), min=1) # topk_anchor_per_gt.sum(1).int():假设该矩阵维度=(3,5)，则把每一行进行相加最后得到一列3行的数据，这样就得到了每个gt对应的候选框最大值
            for gt_id in range(num_gt):
                # TODO：2.通过cost来挑选候选框
                # 这里就相当于给每个gt动态的分配候选框，其中被分配到的候选框的索引会记录到“matching_matrix“矩阵中，对应位置=1
                try:
                    _, res = torch.topk(cost[gt_id], k=dynamic_k[gt_id].item(), largest=False)
                    matching_matrix[gt_id][res] = 1.0
                except Exception as e:
                    print(cost.shape)
                    print(dynamic_k[gt_id].item())
                # _, res = torch.topk(cost[gt_id], k=dynamic_k[gt_id].item(), largest=False)
                # matching_matrix[gt_id][res] = 1.0
            # del topk_anchor_per_gt, dynamic_k, res

            # TODO：3.过滤共用的候选框，即矩阵中同一列有多个1那种，如果有一个候选框被多个”目标“选中，这时候要比较其对应的cost值，较小的值保留候选框
            # TODO：如果存在cost值相等的该怎么办？
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0: # anchor_matching_gt > 1表示存在二义性的候选框
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)# 将cost中，第对应列的值取出，并进行比较，计算最小值所对应的行数，以及分数。
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0 # 将对应位置设置为0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0 # 将对应位置设置为1

            mask_in_boxes = matching_matrix.sum(0) > 0.0
            num_matched_anchors = mask_in_boxes.sum().item()  # 被匹配到的预测框的数量

            anchor_mask[anchor_mask.clone()] = mask_in_boxes #被挑选的预测框对应的位置赋值为True，shape=(num_matched_anchors)

            # bug-test
            matched_gt_index = matching_matrix[:, mask_in_boxes].argmax(0)

            matched_classes_per_gt = gt_classes[matched_gt_index]  # 在真实值中被预测框匹配到的类别，shape（num_matched_anchors）

            matched_ious_per_gt = (matching_matrix * pairwise_iou_approximate).sum(0)[mask_in_boxes]  # 被匹配上的预测框和标注框的iou,shape（num_matched_anchors）

            # 返回匹配到的预测框数量，匹配到的cls, 匹配到的iou,匹配到的gt的index
            return num_matched_anchors, matched_classes_per_gt,matched_ious_per_gt, matched_gt_index, mask_in_boxes

# TODO：将基于Shape的筛选策略修改为“中心点策略”，同时考虑加上额外的惩罚机制
class ComputeLoss_AnchorFree_Decoupled_CenterPoint:
    # Compute losses
    def __init__(self, model, autobalance=False, hyp=None): # add hyp for debug, 2021年11月25日11:19:51
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device # TODO： debug时注销

        # h = model.hyp  # hyperparameters
        h = hyp  # hyperparameters
        # self.class_index = 5 + model.nc
        self.class_index = 5 + 16
        self.model = model  # 复制model，方便后面索引，add

        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)
        BCEcls = nn.BCEWithLogitsLoss(reduction="none")
        BCEobj = nn.BCEWithLogitsLoss(reduction="none")
        BCEangle = nn.BCEWithLogitsLoss(reduction="none").to(device)
        # BCEangle = nn.BCEWithLogitsLoss(reduction="none")


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEangle = FocalLoss(BCEangle, g)
            logging.info("use FocalLoss loss")

        # add, use l1 loss
        # if h['use_l1'] > 0:
        #     BCEcls = nn.L1Loss(reduction="none") # 添加L1 loss
        #     BCEobj = nn.L1Loss(reduction="none") # 添加L1 loss
        #     logging.info("use L1 loss")

        # 获取每一层的output
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEangle = BCEangle
        self.iou_loss = KLDloss()  # add

        # 设置属性
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs=None, img_path=None):  # predictions, targets, model, 增加了两个可视化参数imgs, img_path
        '''
            @param p: list: [small_forward, medium_forward, large_forward]
                eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
            @param targets: shape=(num_gt, [batch_size, class_id, x,y,w,h,theta])
        '''
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # 获取三层feature map分别对应的匹配框的属性
        cls_targets, reg_targets, angle_targets, obj_targets, anchor_masks = self.build_targets_anchor_free(p,
                                                                                                           targets)  # targets

        # 将输出的list进行拼接，每个list包含了batch_size个输出
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        angle_targets = torch.cat(angle_targets, 0)  # add (n_anchor_select_final)
        obj_targets = torch.cat(obj_targets, 0)
        anchor_masks = torch.cat(anchor_masks, 0)

        # 将output也全部torch.cat
        all_outputs = torch.cat(p, 1)
        bbox_preds_with_angle = all_outputs[:, :, :5]  # [batch, anchors_all, 5]
        obj_preds = all_outputs[:, :, 5].unsqueeze(-1)  # [batch, anchors_all, 1]
        cls_preds = all_outputs[:, :, 6:]  # [batch, anchors_all, n_cls]

        reg_targets_with_angle = torch.cat((reg_targets, angle_targets.unsqueeze(-1)), dim=1)

        '''
        1.计算loss，iou部分的loss添加了角度计算
        '''
        # num_gt = max(targets.shape[0],1)
        num_gt = targets.shape[0]
        if num_gt:
            # lbox += (self.iou_loss(reg_targets_with_angle, bbox_preds_with_angle.view(-1, 5)[anchor_masks])).sum() / num_gt # reg_targets_with_angle.shape = (18,5), bbox_preds_with_angle.view(-1, 5).shape = (400,5)
            lbox += (self.giou_loss(p_box=reg_targets_with_angle, t_box=bbox_preds_with_angle.view(-1, 5)[anchor_masks])).sum() / num_gt # reg_targets_with_angle.shape = (18,5), bbox_preds_with_angle.view(-1, 5).shape = (400,5)
            lcls += (self.BCEcls(cls_preds.view(-1, self.nc)[anchor_masks], cls_targets.to(torch.float32))).sum() / num_gt
            lobj += (self.BCEobj(obj_preds.view(-1, 1)[anchor_masks], obj_targets[anchor_masks].float())).sum() / num_gt  # obj_preds[i].shape=(total_anchors,1) obj_targets[i].shape=(total_anchors,1)

            # lobj += (self.BCEobj(obj_preds.view(-1, 1)[anchor_masks], obj_targets[anchor_masks].float())).sum()  # obj_preds[i].shape=(total_anchors,1) obj_targets[i].shape=(total_anchors,1)
            # lbox *= self.hyp['box']
            # lobj *= self.hyp['obj']
            # lcls *= self.hyp['cls']
            # reg_weight = 5.0
            # print("lbox:",lbox)
        # return (reg_weight*lbox + lobj + lcls), torch.cat((reg_weight*lbox, lobj, lcls)).detach()
        return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach()  # 因为 loss不参与更新，所以直接detach()

    # 就类似于get_assignments()方法-yolox
    def build_targets_anchor_free(self, p, targets):
        '''
        @param p: list: [small_forward, medium_forward, large_forward]
            eg: small_forward.size=( batch_size, 1种scale框*size1*size2, [x,y,w,h,theta,obj,classes])
        @param targets: size=(num_gt, [image_id,class,x,y,w,h,theta]), image_id表示当前target属于batch的哪一张
            eg: torch.Size = (num_gt:当前batch中所有目标数量, [该image属于该batch的第几个图片, classId, xywh, Θ])
        '''

        cls_targets = []
        reg_targets = []
        angle_targets = []
        obj_targets = []
        anchor_masks = []

        # 获取预测的输出
        for layer_id, pi in enumerate(p):
            # 得到每一个输出层的属性，一共三层
            # bboxes_preds = pi[:, :, 0:4] # shape = (batch, size1*size2, [x,y,w,h,theta,obj,class])
            # angle_preds = pi[:, :, 4]
            # obj_preds = pi[:, :, 5]
            # cls_preds = pi[:, :, 6:]

            # 逐个batch的处理
            for batch_id in range(pi.shape[0]):
                # TODO:eg：由于targets的第二个维度中的第一列是batch的id，所以需要加一个掩膜，来索引对应batch的GT
                gt_batch_mask = (targets[:, 0] == batch_id)  # gt_batch_mask.shape = (当前batch的num_gt, 7)
                batch_targets = targets[gt_batch_mask]  # 获取某一个batch的GT值
                num_gt_per_batch = batch_targets.shape[
                    0]  # 当前image中GT的数量,shape = (num_gt_per_batch,[class,x,y,w,h,theta])
                batch_pi = pi[batch_id]  # 当前batch中的输出，shape=(1种scale框*size1*size2, [x,y,w,h,theta,obj,classes])

                # 获取某一个batch的输出
                bboxes_preds = batch_pi[:, 0:4]  # shape = (batch, size1*size2, [x,y,w,h,theta,obj,class])

                if num_gt_per_batch:
                    # TODO：初筛操作,anchor_mask.shape=(size1*size2),in_boxes_and_center.shape=(num_gt_per_batch, size1*size2)
                    # in_boxes_and_center: 一个mask，纵轴表示每个GT匹配到的正样本
                    # in_boxes_and_center.sum(1) # 表示同一行全部相加，结果是行的个数，每个位置表示GT有多少个匹配到的Anchor
                    anchor_mask, in_boxes_and_center = self.get_anchor_info(batch_pi, batch_targets, num_gt_per_batch,
                                                                            layer_index=layer_id)
                    get_mask = anchor_mask.unsqueeze(0).repeat(num_gt_per_batch,1)

                    matched_mask = get_mask.sum(0) > 0# 表示被匹配到的anchor的mask矩阵
                    matched_anchor=(get_mask.sum(0) > 0).sum().item()  # 表示被匹配到的anchor的个数
                    matched_cls_index=get_mask.to(torch.int64)[:,matched_mask].argmax(0) # 表示被匹配上的类别id

                    # 获取真实值的信息
                    gt_bboxes = batch_targets[:num_gt_per_batch, 2:6]
                    gt_angles = batch_targets[:num_gt_per_batch, 6]
                    gt_classes = batch_targets[:num_gt_per_batch, 1]  #

                    # 处理匹配到的anchor的各种属性
                    cls_target = F.one_hot(gt_classes[matched_cls_index].to(torch.int64), self.nc)   # (num_select_final, self.nc)
                    obj_target = anchor_mask.unsqueeze(-1)  # （total_num_anchor, 1）
                    # obj_target = mask_in_boxes.unsqueeze(-1)  # （n_anchor_select_final, 1）
                    reg_target = gt_bboxes[matched_cls_index]  # (n_anchor_select_final, 4)
                    angle_target = gt_angles[matched_cls_index]  # add  （n_anchor_select_final）

                    # 直接用初步筛选得到的正样本进行回归和分类
                else:
                    cls_target = pi.new_zeros((0, self.nc))
                    reg_target = pi.new_zeros((0, 4))
                    angle_target = pi.new_zeros(0)
                    obj_target = pi.new_zeros((pi.shape[1], 1))
                    anchor_mask = pi.new_zeros(pi.shape[1]).bool()
                cls_targets.append(cls_target)
                reg_targets.append(reg_target)
                angle_targets.append(angle_target)  # add
                obj_targets.append(obj_target)
                anchor_masks.append(anchor_mask)

        return cls_targets, reg_targets, angle_targets, obj_targets, anchor_masks

    '''
    功能：初步筛选正样本的anchor，为精细化筛选做准备
    '''
    def get_anchor_info(self, pi, targets, num_gt_per_batch, layer_index):
        '''
        @param pi: 传入的是batch中某一个的特征图，size = (1种scale框 * size1 * size2, [x,y,w,h,theta,obj,classes])
            eg: 将检测头的每一层输出的特征图w，h进行合并
        @param targets: 当前batch中某一个image的targets，shape=(num_gt_per_batch, [class,x,y,w,h,theta])
        '''
        # exp_strides = []  # 保存采样率
        strides = [6, 16, 32]  # 三层layer的采样率
        num_anchor, num_gt_per_batch = self.na, num_gt_per_batch  # 预测框的种类, 当前image中标签值的数量

        # 返回所有符合条件的正样本anchor，它是一个list，size=3
        boxes_or_center = []
        boxes_and_center = []

        # 初始化网格空间的缩放增益
        gain = torch.ones(7, device=targets.device)
        xx = [1, int(np.sqrt(pi.shape[0])), int(np.sqrt(pi.shape[0])), 1]  # (w,h,22)
        gain[2:6] = torch.tensor(xx)[[2, 1, 2, 1]]  # 特征图的w,h,w,h

        # targets: size=(num_anchor, num_gt,[image,class_id,x,y,w,h,theta])
        t = targets * gain  # 将标签值缩放到特征图上

        # 所有的anchor数量
        total_num_anchors = pi.shape[0]

        # 获取grid的x,y坐标
        if RANK != -1:  # DDP
            x_shifts = self.model.module.model[-1].grid[layer_index][:, :, 0]
            y_shifts = self.model.module.model[-1].grid[layer_index][:, :, 1]
        else:
            x_shifts = self.model.model[-1].grid[layer_index][:, :, 0]  # shape = (1, size1*size2)
            y_shifts = self.model.model[-1].grid[layer_index][:, :, 1]

        # exp_stride: 保存每一层的采样间隔，size=(1, grid.shape[1])
        exp_strides = torch.zeros(1, pi.shape[0]).fill_(strides[layer_index]).type_as(pi[0])

        # 获取每个特征图的x,y格点坐标
        # x_shifts_per_img = x_shifts * exp_strides[layer_index]
        # y_shifts_per_img = y_shifts * exp_strides[layer_index]
        x_shifts_per_img = x_shifts[0] * exp_strides
        y_shifts_per_img = y_shifts[0] * exp_strides

        # 获取每个格点的中心坐标,左上角为原点
        # x_c_per_img.shape = [num_gt_per_batch, total_num_anchors]
        x_c_per_img = ((x_shifts_per_img + 0.5 * exp_strides).repeat(num_gt_per_batch, 1))
        y_c_per_img = ((y_shifts_per_img + 0.5 * exp_strides).repeat(num_gt_per_batch, 1))

        # TODO：初步筛选
        # 计算标签的左上角和右下角坐标，即（[left,top], [right,bottom])
        # gt_xxx与x_c_per_img维度应该相同
        # gt_l = ( (t[: , :, 2] - 0.5 * t[:, :, 4]).squeeze(1).repeat(1,total_num_anchors))
        # gt_t = ( (t[: , :, 3] - 0.5 * t[:, :, 5]).squeeze(1).repeat(1,total_num_anchors))
        # gt_r = ( (t[: , :, 2] + 0.5 * t[:, :, 4]).squeeze(1).repeat(1,total_num_anchors))
        # gt_b = ( (t[: , :, 3] + 0.5 * t[:, :, 5]).squeeze(1).repeat(1,total_num_anchors))
        gt_l = ((t[:, 2] - 0.5 * t[:, 4]).unsqueeze(1).repeat(1, total_num_anchors))  # shape=(num_gt_per_img, total_anchors)
        gt_t = ((t[:, 3] - 0.5 * t[:, 5]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_r = ((t[:, 2] + 0.5 * t[:, 4]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_b = ((t[:, 3] + 0.5 * t[:, 5]).unsqueeze(1).repeat(1, total_num_anchors))

        # TODO ：anchor free的思路是每个格点都作为一个预测的anchor，因此anchor数量就是当前的特征图大小;
        # 计算出每个anchor的左上角和右上角坐标（左上角为原点），然后与真实的标签值进行判断“当前anchor是否处于GT的内部”：如果是则为正样本
        bbox_l = x_c_per_img - gt_l
        bbox_r = gt_r - x_c_per_img
        bbox_t = y_c_per_img - gt_t
        bbox_b = gt_b - y_c_per_img
        bboxes = torch.stack([bbox_l, bbox_t, bbox_r, bbox_b], 2)  # size = (num_gt, feature_size1*feature_size2,4)

        # 然后将所有落在GT中的anchor挑选出来
        in_boxes = bboxes.min(dim=-1).values > 0.0  # 必须全部大于0才是需要的anchor
        in_boxes_all = in_boxes.sum(dim=0) > 0  # 中心点位于标注框内的锚框为True,相当于一个mask

        # TODO 再次筛选：绘制一个边长为5的正方形。左上角点为（gt_l，gt_t），右下角点为（gt_r，gt_b）。gt以正方形范围形式去挑选锚框
        radius = 2  # 半径
        # gt_l = (targets[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides[i]# x - radius*stride
        # gt_r = (targets[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[i]
        # gt_t = (targets[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides[i]
        # gt_b = (targets[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[i]
        gt_l = (t[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides  # x - radius*stride
        gt_t = (t[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides
        gt_r = (t[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides
        gt_b = (t[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides
        # gt_b = (t[:, :, 3].squeeze(0).unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[layer_index]

        c_l = x_c_per_img - gt_l
        c_r = gt_r - x_c_per_img
        c_t = y_c_per_img - gt_t
        c_b = gt_b - y_c_per_img
        center = torch.stack([c_l, c_t, c_r, c_b], 2)
        in_centers = center.min(dim=-1).values > 0.0
        in_centers_all = in_centers.sum(dim=0) > 0

        # 某一边在gt里面
        # boxes_or_center.append(in_boxes_all | in_centers_all)
        boxes_or_center = in_boxes_all | in_centers_all  # 非 list版本

        # 两者都在gt里面
        # boxes_and_center.append(in_boxes[:, boxes_or_center[i]] & in_centers[:, boxes_or_center[i]])
        boxes_and_center = in_boxes[:, boxes_or_center] & in_centers[:, boxes_or_center]  # 非 list版本

        return boxes_or_center, boxes_and_center

    # 使用GIou loss来计算旋转框的LOSS
    def giou_loss(self, p_box, t_box, x1y1x2y2=False, GIoU=True,eps=1e-7):
        # p_box1 [[x,y,w,h,angle], n]
        assert p_box.shape[0] == t_box.shape[0]
        N = p_box.shape[0]
        M = t_box.shape[0]
        overlaps = np.zeros((N,M),dtype=np.float32)
        # p_box = p_box.T # [n,5]->[5,n]
        # t_box = t_box.T # [m,5]->[5,m]

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = p_box
            p_x1, p_y1, p_x2, p_y2 = p_box[0], p_box[1], p_box[2], p_box[3]
            t_x1, t_y1, t_x2, t_y2 = t_box[0], t_box[1], t_box[2], t_box[3]
        else:
            # TODO: x,y,w,h,angle转化为x1,y1,x2,y2,左上角和右下角
            p_x1, p_x2 = p_box[0] - p_box[2] / 2, p_box[0] + p_box[2] / 2
            p_y1, p_y2 = p_box[1] - p_box[3] / 2, p_box[1] + p_box[3] / 2
            t_x1, t_x2 = t_box[0] - t_box[2] / 2, t_box[0] + t_box[2] / 2
            t_y1, t_y2 = t_box[1] - t_box[3] / 2, t_box[1] + t_box[3] / 2

        iou = None
        union = None

        for i in range(N):
            p_area = p_box[i][2] * p_box[i][3]
            for j in range(M):
                t_area = t_box[j][2] * t_box[j][3]
                union = p_area + t_area + eps

                box1 = ((p_box[i][0].item(),p_box[i][1].item()), (p_box[i][2].item(),p_box[i][3].item()), (p_box[i][4].item()))
                box2 = ((t_box[j][0].item(),t_box[j][1].item()), (t_box[j][2].item(),t_box[j][3].item()), (t_box[j][4].item()))

                # 计算两个rectangle之间是否有交集
                # 如果有交集，那么会返回交集的所有相交的顶点即 contours
                flag, contours = cv2.rotatedRectangleIntersection(box1, box2)

                if flag ==1:
                    inter = np.round(np.abs(cv2.contourArea(contours)))
                    overlaps[i,j] = inter / (union - inter) # 计算交并比

                if flag ==2:
                    inter = np.minimum(union-t_area, t_area)
                    overlaps[i,j] = inter / (union - inter) # 计算交并比

        # cv2.boxPoints(box1) # xywha形式的box返回四个顶点


        # 计算两个旋转框的IoU
        # inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
        #         (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
        #
        # # Union Area
        # w1, h1 = p_x2 - p_x1, p_y2 - p_y1 + eps
        # w2, h2 = t_x2 - t_x1, t_y2 - t_y1 + eps
        # union = w1 * h1 + w2 * h2 - inter + eps


        if GIoU :
            cw = torch.max(p_x2, t_x2) - torch.min(p_x1, t_x1)  # convex (smallest enclosing box) width
            ch = torch.max(p_y2, t_y2) - torch.min(p_y1, t_y1)  # convex height
             # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU

class ComputeLoss_AnchorFree_FCOS:
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        self.class_index = 5 + model.nc
        self.model = model  # 复制model，方便后面索引，add

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEangle = FocalLoss(BCEangle, g)

        # 获取每一层的output
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEangle = BCEangle
        self.iou_loss = KLDloss()  # add

        # 设置属性
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs=None, img_path=None):  # predictions, targets, model, 增加了两个可视化参数imgs, img_path
        '''
            @param p: list: [small_forward, medium_forward, large_forward]
                eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
            @param targets: shape=(num_gt, [batch_size, class_id, x,y,w,h,theta])
        '''
        device = targets.device

    def build_targets(self,p,targets):
        pass
        '''
            
        '''

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s ', level=logging.INFO)

    """
            yolov5l: Model Summary: 499 layers, 48026065 parameters, 48026065 gradients, 118.7 GFLOPs
            yolov5m: Model Summary: 391 layers, 22103025 parameters, 22103025 gradients, 53.8 GFLOPs;
            yolov5s: Model Summary: 283 layers, 7762065 parameters, 7762065 gradients, 18.6 GFLOPs;
            yolov5x: Model Summary: 607 layers, 88987185 parameters, 88987185 gradients, 222.9 GFLOPs
            yolov5m-asff: 10948910
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../models/yolov5m.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='yolov5m-anchor-free-decoupled.yaml', help='model.yaml')
    parser.add_argument('--hyp', type=str,
                        default='..//data/hyps/hyp.scratch.yaml',
                        # default=os.path.dirname(os.path.abspath(__file__)) + '/data/hyps/hyp.finetune_anchor_free.yaml',
                        # default=os.path.dirname(os.path.abspath(__file__)) + '/data/hyps/hyp.scratch_anchor_free.yaml',
                        help='hyperparameters path')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    device = select_device(opt.device)
    logging.info(opt.cfg)

    # Create model
    model = Model(opt.cfg, ch=3, nc=16, anchors=hyp.get('anchors')).to(device)
    logging.info("loaded model done!")

    loss = ComputeLoss_AnchorFree_Decoupled_CenterPoint(model,hyp = hyp)



