# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)
import cv2
from torchvision.utils import save_image # 特征图可视化工具，2021-10-17 17:07:35

# 原始检测头
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=16, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180  # number of outputs per anchor, 180 mean angle classifications.
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors，3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # y: torch.Size([1, 1, 4, 4, 265])
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

#-------------------------------------------------------------------------#
# 选择不同激活函数
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
# 构建基础卷积
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
'''
    将传统的检测头几个任务进行解耦，一般用1x1卷积进行通道降维，然后再接3*3卷积:
        回归分支再解耦：box（H*W*4） + object（H*W*1）；
        分类分支：class（（H*W*C）；
'''
class Detect_AnchorFree_Decoupled(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export,网络模型输出为onnx格式，可在其他深度学习框架上运行
    '''
        nc: 类别数
        ch: 通道数=3
        anchors: anchor数量（anchor free都是1）
    '''
    def __init__(self, nc=16, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        '''
            1.anchor based-> anchor free：（nc + 5 + angle) -> (num_classes + x,y,left,top,right,bottom,score)
            2.有了anchor free后就不需要每种anchor设置3种scale；
        '''
        self.nc = nc  # number of classes
        # self.no = nc + 7 # 每一个anchor需要预测的维度数（num_classes + x,y,left,top, right,bottom + score）
        self.no = nc + 5 + 1 # 每一个anchor需要预测的维度数（num_classes + x,y,w,h,score + theta）
        # self.nl = 3  # number of detection layers  3
        self.nl = len(anchors)  # 三种步长的检测网络
        self.na = len(anchors[0]) // 2  # 每一个anchor的种类，anchor free只有1种anchor

        # 缓存对应的anchor
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl * self.na, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.grid = [torch.zeros(1)] * self.nl  # init grid   [tensor([0.]), tensor([0.]), tensor([0.])] 初始化网格

        # 对【self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)】”分类“和”回归“任务进行解耦
        self.cls_conv = nn.ModuleList() # 分类卷积
        self.reg_conv = nn.ModuleList() # 回归卷积

        self.act = 'relu'  # 激活函数形式
        self.cls_pred = nn.ModuleList() # 一个1x1的卷积，把通道数变成类别数，比如coco 16类
        self.reg_pred = nn.ModuleList() # 一个1x1的卷积，把通道数变成4通道，xywh.
        self.obj_pred = nn.ModuleList() # 一个1x1的卷积，把通道数变成1通道，obj
        self.angle_preds = nn.ModuleList() # angle，预测一个通道
        self.stems = nn.ModuleList() # 进行通道压缩的卷积

        # 3种不同尺度的输出进行初始化卷积
        # 同时生成grid，为build_target_anchor_free处理正样本anchor做准备
        for i in range(len(ch)):
            # 构建1*1卷积进行通道降维
            self.stems.append(BaseConv(in_channels=(int(ch[i])), out_channels=int(ch[i]),ksize=1,stride=1,act=self.act))

            # 构建分类和回归任务的3*3卷积，这里只使用一层BaseConv，考虑复杂度的问题
            self.cls_conv.append(nn.Sequential(*[
                BaseConv(in_channels=(int(ch[i])), out_channels=int(ch[i]),ksize=3,stride=1,act=self.act)
                # BaseConv(in_channels=(int(ch[i])), out_channels=int(ch[i]),ksize=3,stride=1,act=self.act),
            ]))
            self.reg_conv.append(nn.Sequential(*[
                BaseConv(in_channels=(int(ch[i])), out_channels=int(ch[i]), ksize=3, stride=1, act=self.act)
                # BaseConv(in_channels=(int(ch[i])), out_channels=int(ch[i]), ksize=3, stride=1, act=self.act),
            ]))

            # 构建预测的卷积
            self.cls_pred.append(nn.Conv2d(in_channels=int(ch[i]), out_channels=self.na * self.nc, kernel_size=1,stride=1))
            self.reg_pred.append(nn.Conv2d(in_channels=int(ch[i]), out_channels=4, kernel_size=1,stride=1))
            self.obj_pred.append(nn.Conv2d(in_channels=int(ch[i]), out_channels=self.na * 1, kernel_size=1,stride=1))
            self.angle_preds.append(nn.Conv2d(in_channels=int(ch[i]), out_channels=1, kernel_size=1,stride=1))



    def forward(self, x):
        '''
        相当于最后生成的feature map分辨率为size1 × size2.即映射到原图，有size1 × size2个锚点，以锚点为中心生成锚框来获取Region proposals，每个锚点代表一个[xywh,score,num_classes]向量
            :arg x:
                [(P3/8-small), (P4/16-medium), (P5/32-large)]   (3种size的featuremap, batch_size, no * na , size_1, size2)
            :return:
                 if training :
                    x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
                 else : (z,x)
                    z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), no) 真实坐标
                    x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
        '''
        # x = x.copy()  # for profiling

        z = []  # inference output
        self.training |= self.export

        for i, (cls_conv, reg_conv, xi) in enumerate(
                zip(self.cls_conv, self.reg_conv, x)):
            '''
            先进行通道压缩
            '''
            xi= self.stems[i](xi) # 依次处理三层feature

            # 分别进行解耦卷积
            cls_x = xi
            reg_x = xi

            # 分类的输出
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_pred[i](cls_feat)

            # 回归的输出
            reg_feat = cls_conv(reg_x)
            reg_output = self.reg_pred[i](reg_feat) # 坐标的输出
            obj_output = self.obj_pred[i](reg_feat) # obj的输出

            # 角度的输出
            angle_output = self.angle_preds[i](reg_feat) # 角度的输出同样用到回归的数据

            # 对每一层的结果进行拼接
            '''
                reg_output=()
            '''
            x[i] = torch.cat([reg_output, angle_output, obj_output, cls_output], 1)
            # print(x[i].shape)

            bs, _, ny, nx = x[i].shape  # x[i]:(batch_size, (5+nc) * na, size1', size2')

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 构建grid
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                grid = self._make_grid(nx, ny).to(x[i].device)
                self.grid[i] = grid.view(1,-1,2) # size=(1,h_size*w_size,2)
            # inference推理模式,相当于解码操作
            # if not self.training:
            #     # 以height为y轴，width为x轴的grid坐标 坐标按顺序（0, 0） （1, 0）... (width-1, 0) (0, 1) (1,1) ... (width-1, 1) ... (width-1 , height-1)
            #     if self.grid[i].shape[2:4] != x[i].shape[2:4]:
            #         self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            #
            #     y = x[i].sigmoid() # 归一化处理
            #
            #     # xy 预测的真实坐标 y[..., 0:2] * 2. - 0.5是相对于左上角网格的偏移量； self.grid[i]是网格坐标索引
            #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
            #
            #     # wh 预测的真实wh  self.anchor_grid[i]是原始anchors宽高  (y[..., 2:4] * 2) ** 2 是预测出的anchors的wh倍率
            #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            #     z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):  # 绘制网格
        """
        绘制网格 eg：640 × 480的图像在detect层第一层中featuremap大小为 80 × 60，此时要生成 80 × 60的网格在原图上
        @param nx: 当前featuremap的width
        @param ny: 当前featuremap的height
        @return: tensor.shape(1, 1, 当前featuremap的height, 当前featuremap的width, 2) 生成以height为y轴，width为x轴的grid坐标
                 坐标按顺序（0, 0） （1, 0）...  (width-1, 0) (0, 1) (1,1) ... (width-1, 1) ... (width-1 , height-1)
        """
        # 初始化ny行 × nx列的tensor
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])

        # 将两个 ny×ny 和 nx×nx的tensor在dim=2的维度上进行堆叠 shape(ny, nx, 2)
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

'''
    修改为anchor-free的耦合头
'''
class Detect_AnchorFree(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=1, ch=(), inplace=True):  # detection layer
        super().__init__()
        '''
            1.采用了AnchorFree后，每个cell只预测一个框，因此anchor的种类=1；
            2.对于角度问题，每一层的输出增加了180个角度的分类值(原本的YOLO是耦合到每一层的输出)，这里考虑解耦到单独的一个卷积进行预测
        '''
        self.nc = nc  # number of classes
        self.no = nc + 5# number of outputs per anchor,[x,y,w,h,score,angle]
        self.nl = len(ch)  # number of detection layers
        self.na = 1  # 每一层的每一个cell只预测一种anchor，这样就避免预设anchor值
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 构建检测头的三层输出卷积
        self.angle_preds = nn.ModuleList(nn.Conv2d(in_channels=x,out_channels=1,kernel_size=1) for x in ch)# 构建角度预测的卷积
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output

        for i in range(self.nl):
            angle_feat = x[i] # add
            angle_output = self.angle_preds[i](angle_feat) # add
            x[i] = self.m[i](x[i])  # conv
            # x[i] = torch.cat([x[i],angle_output],1) # 将角度的预测结果与原结果进行拼接

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



#----------------------------------------------------#
"""
    功能：基于ASFF的检测头
    时间：2021-10-13 15:46:54
"""
class ASFF_Detect(nn.Module):   #add ASFFV5 layer and Rfb
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), multiplier=0.5,rfb=False,ch=(),inplace=True):  # detection layer
        super(ASFF_Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180 # 旋转目标检测需要添加‘180’angle的维度
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.l0_fusion = ASFFV5(level=0, multiplier=multiplier,rfb=rfb)
        self.l1_fusion = ASFFV5(level=1, multiplier=multiplier,rfb=rfb)
        self.l2_fusion = ASFFV5(level=2, multiplier=multiplier,rfb=rfb)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        result=[]
        # self.training |= self.export
        result.append(self.l2_fusion(x))
        result.append(self.l1_fusion(x))
        result.append(self.l0_fusion(x))
        x=result
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # env for windows,2021年10月27日10:00:19
            with open(cfg,encoding='UTF-8') as f:
                self.yaml = yaml.safe_load(f)  # model dict
            # with open(cfg) as f:
            #     self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors---"检测头“
        m = self.model[-1]  # Detect()

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            print('Strides: %s' % m.stride.tolist())

        elif isinstance(m, ASFF_Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            print('Strides: %s' % m.stride.tolist())

        # 构建anchor free的检测头
        elif isinstance(m, Detect_AnchorFree_Decoupled):
            s = 128
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # m.anchors /= m.stride.view(-1, 1, 1)
            # check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
            print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info() # 这里计算了模型的复杂度
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None

        return self.forward_once(x, profile, visualize)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # 保存特征图
            # print("save image")
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        '''
        网络的整体前向传播计算（backbone+neck+detect）
            @param x: 待前向传播到的向量，shape(batch_size，3，height，width)
            @param profile：是否进行复杂度FLOPs计算；
            @param visualize：是否进行可视化
            @return：
                if training：x list[small_forward, medium_forward, large_forward], eg: small_forward=(batch_size, anchor种类, size1,size2,[xywh,score,num_classes,num_angle])
                else: (z,x)
                    z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), no)
                    x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
            TODO：由于anchor base改为了anchor free，那么上述返回的list中的size大小中的num_angle该如何变化？
        '''
        y, dt = [], []  # outputs
        for m in self.model:
            # 如果不是来自上一层的卷积结果
            if m.f != -1:
                # 这里的x一般为concat/Detect的前向计算结果
                # 例如：m=Concat, m.f=[-1,14],x=[x,y[14]],即x=[上一层的计算结果，第14层的计算结果]
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # 复杂度计算
            if profile:
                c = isinstance(m, Detect_AnchorFree_Decoupled)  # copy input as inplace fix
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_sync()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            # 进行前向传播计算
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            # 可视化特征图，2021-10-17 17:26:41
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=Path('runs/visualization'))

        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        # add
        if isinstance(m,Detect):
            #原代码, 耦合头部分的偏置初始化变量
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # add
        if isinstance(m, Detect_AnchorFree_Decoupled):
            # add, 因为修改了检测头中的m卷积层（耦合头变为解耦头）
            for mi, s in zip(m.cls_pred, m.stride):
                # cls
                b = mi.bias.view(m.na, -1)
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            for mj, s in zip(m.obj_pred, m.stride): # add,obj
                b= mj.bias.view(m.na, -1)
                b.data.fill_(-math.log((1 - 1e-2) / 1e-2)) # 初始化偏置
                mj.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # add,初始化角度的偏置变量
            for mk in m.angle_preds:
                b = mk.bias.view(1, -1)
                b.data.fill_(-math.log((1 - 1e-2) / 1e-2))
                mk.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:

            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum([ch[x] for x in f])

        elif m is Detect: # 检测头
            # args=(80,[]
            args.append([ch[x] for x in f])

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        # 增加ASFF的检测头,2021年11月1日10:46:01
        elif m is ASFF_Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        # 增加AnchorFree的【耦合】检测头，2021年11月1日10:46:04
        elif m is Detect_AnchorFree:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        # 增加AnchorFree的【解耦】检测头，2021年11月1日10:46:04
        elif m is Detect_AnchorFree_Decoupled:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    """
        yolov5l: Model Summary: 499 layers, 48026065 parameters, 48026065 gradients, 118.7 GFLOPs
        yolov5m: Model Summary: 391 layers, 22103025 parameters, 22103025 gradients, 53.8 GFLOPs;
        yolov5s: Model Summary: 283 layers, 7762065 parameters, 7762065 gradients, 18.6 GFLOPs;
        yolov5x: Model Summary: 607 layers, 88987185 parameters, 88987185 gradients, 222.9 GFLOPs
        yolov5m-asff: 10948910
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5m.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    set_logging()
    device = select_device(opt.device)
    print(opt.cfg)

    # Create model
    model = Model(opt.cfg).to(device)
    # # model.train()
    x = torch.rand(1,3,224,224).to(device)
    # res = model(x)
    #
    # for x in res:
    #     print(x.shape)
    flops, params = profile(model, inputs=(x,))
    print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops / 1e9, params / 1e6))
    # exit(1)
    # Profile
    # if opt.profile:
    #     img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    #     y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
