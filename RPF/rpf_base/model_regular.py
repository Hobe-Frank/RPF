import os# 增加下载超时时间（默认 10 秒太短，改为 60 秒）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# 可选：启用离线重试优化
os.environ["HF_HUB_ENABLE_HTTP_STREAMING"] = "1"  # 支持流式下载大文件
import torch
import timm
import torch.nn as nn
from rpf_base import helper
from rpf_base.aggregators.mixvpr import VarianceMixVPR
import torch.nn.functional as F

class VPRModel(nn.Module):  # 继承pytorch-lightning.LightningModule模块
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone
                 model_name='dinov2_vitb14_MixVPR',
                 backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,

                 # ---- Aggregator
                 agg_arch='MixVPR',
                 agg_config={},

                 ):
        super().__init__()
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.agg_config = agg_config
        # self.save_hyperparameters()  # write hyperparams into a file
        self.model_name = model_name
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1=self.layer1, use_cls=self.use_cls,
                                            norm_descs=self.norm_descs)
        self.mean = helper.get_aggregator(agg_arch, agg_config)
        self.std = VarianceMixVPR(**agg_config)

    # the forward pass of the lightning model
    def forward(self, x,state='mean'):
        x = self.backbone(x)
        feat_mean = self.mean(x)
        if state=='std':
            feat_std = self.std(x)
        if state=='std':
            return feat_mean,feat_std
        return feat_mean

class TimmModel(nn.Module):

    def __init__(self,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 layer1=8,
                 # Aggregator
                 agg_arch='MixVPR',
                 agg_config={},
                 ):

        super(TimmModel, self).__init__()

        self.img_size = img_size

        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, agg_arch=agg_arch, layer1=layer1, agg_config=agg_config)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # self.regular = Regular()
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = torch.nn.Parameter(torch.ones([]))
        if pretrained_path:
            # 加载预训练模型的权重，但不包括输出层的权重
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)

    def get_config(self):

        data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None,state='mean'):
        if img2 is not None:

            image_features1 = self.model(img1,state=state)
            image_features2 = self.model(img2,state=state)

            return image_features1, image_features2

        else:
            image_features = self.model(img1,state=state)

            return image_features

