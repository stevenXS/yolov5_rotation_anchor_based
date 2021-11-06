# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np
import math

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from utils.general import *
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

        #可视化target，2021-10-18 14:28:36
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
        @param targets : torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, Θ])
        @param model : 模型
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

            if nt:
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
                t = t.repeat((5, 1, 1))[j]

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
            gij = (gxy - offsets).long() # 获取每个box所在网格点的坐标
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 7].long()  # 每个anchor的索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 保存图片序号、anchor索引、网格点坐标
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 获取xy相对于网格点的偏置，以及box宽高
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tangle.append(angle)  # angle

        return tcls, tangle, tbox, indices, anch


class ComputeLoss_AnchorFree:
        # Compute losses
        def __init__(self, model, autobalance=False):
            self.sort_obj_iou = False
            device = next(model.parameters()).device  # get model device
            h = model.hyp  # hyperparameters
            self.class_index = 5 + model.nc
            self.model = model

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
            '''
            @param p: list: [small_forward, medium_forward, large_forward]
                eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
            '''
            device = targets.device
            lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1,
                                                                                                         device=device)
            langle = torch.zeros(1, device=device)

            # build_targets函数返回的结果应该是正样本
            boxes_or_center, boxes_and_center = self.build_targets_anchor_free(p, targets)  # targets

            # pi.size= (batch_size,channel, feature_map_size1,feature_map_size2, [x,y,w,h,obj]+ num_classes +180angle)
            for i, pi in enumerate(p):  # layer index, layer predictions

                bboxes_preds = pi[:,:,:, :4] # x,y,w,h
                angle_preds = pi[:,:,:, 4]  # theta
                cls_preds = pi[:,:,: 5: self.class_index]


            lbox *= 1
            lobj *= 1
            lcls *= 1
            langle *= 1
            bs = 1  # batch size

            return (lbox + lobj + lcls + langle) * bs, torch.cat((lbox, lobj, lcls, langle)).detach()


        def build_targets_anchor_free(self, p, targets):
            '''
            @param p: list: [small_forward, medium_forward, large_forward]
                eg:small_forward.size=( batch_size, 1种scale框, size1, size2, [class,x,y,w,h,theta])
            @param targets: size=(image,class,x,y,w,h,theta)
            '''
            x_shifts = [] # grid的x坐标
            y_shifts = [] # grid的y坐标
            exp_strides = [] # 保存采样率
            strides = [8,16,32] # 三层layer的采样率

            num_anchor, num_gt = self.na, targets.shape[0]  # 预测框的数量, 标签值的数量

            # 返回所有符合条件的正样本anchor，它是一个list，size=3
            boxes_or_center = []
            boxes_and_center = []


            # 保持和原来的build——target返回的结果一直
            tcls, tbox, tangle= [], [], []
            indices, anch = [], []

            for i in range(self.nl):
                # 所有的anchor数量
                total_num_anchors = self.model.model[-1].grid[i].shape[1]

                # 获取grid的x,y坐标
                x_shifts.append(self.model.model[-1].grid[i][:, :, 0]) # 假如特征图h=20,则有20组0···19
                y_shifts.append(self.model.model[-1].grid[i][:, :, 1]) # 假如特征图h=20,则分别有20个0···19

                # exp_stride: 保存每一层的采样间隔，size=(1, grid.shape[1])
                exp_strides.append(
                    torch.zeros(1,self.model.model[-1].grid[i].shape[1])
                    .fill_(strides[i])
                    .type_as(p[0])
                )

                # 获取每个特征图的x,y格点坐标
                x_shifts_per_img = x_shifts[i] * exp_strides[i]
                y_shifts_per_img = y_shifts[i] * exp_strides[i]

                # 获取每个格点的中心坐标,左上角为原点
                x_c_per_img = ((x_shifts_per_img + 0.5*exp_strides[i]).repeat(num_gt,1)) # [num_anchors] ->[num_gt,num_anchors]
                y_c_per_img = ((y_shifts_per_img + 0.5*exp_strides[i]).repeat(num_gt,1))


                # TODO：初步筛选
                # 计算标签的左上角和右下角坐标，即（[left,top], [right,bottom])
                # gt_xxx与x_c_per_img维度应该相同
                gt_l = ( (targets[: , 2] - 0.5 * targets[:, 4]).unsqueeze(1).repeat(1,total_num_anchors))
                gt_t = ( (targets[: , 3] - 0.5 * targets[:, 5]).unsqueeze(1).repeat(1,total_num_anchors))
                gt_r = ( (targets[: , 2] + 0.5 * targets[:, 4]).unsqueeze(1).repeat(1,total_num_anchors))
                gt_b = ( (targets[: , 3] + 0.5 * targets[:, 5]).unsqueeze(1).repeat(1,total_num_anchors))


                # TODO ：anchor free的思路是每个格点都作为一个预测的anchor，因此anchor数量就是当前的特征图大小;
                # 计算出每个anchor的左上角和右上角坐标（左上角为原点），然后与真实的标签值进行判断“当前anchor是否处于GT的内部”：如果是则为正样本
                bbox_l = x_c_per_img - gt_l
                bbox_r = gt_r - x_c_per_img
                bbox_t = y_c_per_img - gt_t
                bbox_b = gt_b - y_c_per_img
                bboxes = torch.stack([bbox_l,bbox_t,bbox_r,bbox_b], 2) # size = (num_gt, feature_size1*feature_size2,4)

                # 然后将所有落在GT中的anchor挑选出来
                in_boxes = bboxes.min(dim=-1).values>0.0 # 必须全部大于0才是需要的anchor
                in_boxes_all = in_boxes.sum(dim=0) > 0 # 中心点位于标注框内的锚框为True,相当于一个mask

                # TODO 再次筛选：绘制一个边长为5的正方形。左上角点为（gt_l，gt_t），右下角点为（gt_r，gt_b）。gt以正方形范围形式去挑选锚框
                radius = 2.5 # 半径

                gt_l = (targets[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides[i]# x - radius*stride
                gt_r = (targets[:, 2].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[i]
                gt_t = (targets[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) - radius * exp_strides[i]
                gt_b = (targets[:, 3].unsqueeze(1).repeat(1, total_num_anchors)) + radius * exp_strides[i]

                c_l = x_c_per_img - gt_l
                c_r = gt_r - x_c_per_img
                c_t = y_c_per_img - gt_t
                c_b = gt_b - y_c_per_img
                center = torch.stack([c_l, c_t, c_r, c_b], 2)
                in_centers = center.min(dim=-1).values > 0.0
                in_centers_all = in_centers.sum(dim=0) > 0

                # 某一边在gt里面
                boxes_or_center.append(in_boxes_all | in_centers_all)
                # 两者都在gt里面
                boxes_and_center.append(in_boxes[:, boxes_or_center[i]] & in_centers[:, boxes_or_center[i]])

                # Append
                b, c = boxes_and_center[:, :2].long().T  # 获取每个标签的图像，类别索引，第0,1维度
                a = boxes_and_center[i][:, 7].long()  # 每个anchor的索引


            return boxes_or_center, boxes_and_center

































