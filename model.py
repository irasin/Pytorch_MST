import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_transfer import MultimodalStyleTransfer
from normalisedVGG import NormalisedVGG
from VGGdecoder import Decoder
from utils import download_file_from_google_drive


def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1)
    return features_mean, features_std


class VGGEncoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        vgg = NormalisedVGG(pretrained_path=pretrained_path).net
        self.block1 = vgg[: 4]
        self.block2 = vgg[4: 11]
        self.block3 = vgg[11: 18]
        self.block4 = vgg[18: 31]

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=True):
        h1 = self.block1(images)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class Model(nn.Module):
    def __init__(self,
                 n_cluster=3,
                 alpha=1,
                 device='cpu',
                 lam=0.1,
                 pre_train=False,
                 max_cycles=None):
        super().__init__()
        self.n_cluster = n_cluster
        self.alpha = alpha
        self.device = device
        self.lam = lam
        self.max_cycles = max_cycles
        if pre_train:
            if not os.path.exists('vgg_normalised_conv5_1.pth'):
                download_file_from_google_drive('1IAOFF5rDkVei035228Qp35hcTnliyMol',
                                                'vgg_normalised_conv5_1.pth')
            if not os.path.exists('decoder_relu4_1.pth'):
                download_file_from_google_drive('1kkoyNwRup9y5GT1mPbsZ_7WPQO9qB7ZZ',
                                                'decoder_relu4_1.pth')
            self.vgg_encoder = VGGEncoder('vgg_normalised_conv5_1.pth')
            self.decoder = Decoder(4, 'decoder_relu4_1.pth')
        else:
            self.vgg_encoder = VGGEncoder()
            self.decoder = Decoder(4)

        self.multimodal_style_feature_transfer = MultimodalStyleTransfer(n_cluster,
                                                                         alpha,
                                                                         device,
                                                                         lam,
                                                                         max_cycles)

    def generate(self,
                 content_images,
                 style_images,
                 n_cluster=None,
                 alpha=None,
                 device=None,
                 lam=None,
                 max_cycles=None):

        n_cluster = self.n_cluster if n_cluster is None else n_cluster
        alpha = self.alpha if alpha is None else alpha
        device = self.device if device is None else device
        lam = self.lam if lam is None else lam
        max_cycles = self.max_cycles if max_cycles is None else max_cycles

        multimodal_style_feature_transfer = MultimodalStyleTransfer(n_cluster,
                                                                    alpha,
                                                                    device,
                                                                    lam,
                                                                    max_cycles)

        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        cs = []

        for c, s in zip(content_features, style_features):
            cs.append(multimodal_style_feature_transfer.transfer(c, s).unsqueeze(dim=0))
        cs = torch.cat(cs, dim=0)

        out = self.decoder(cs)
        return out

    @staticmethod
    def calc_content_loss(out_features, content_features):
        return F.mse_loss(out_features, content_features)

    @staticmethod
    def calc_style_loss(out_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(out_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, style_images, gamma=1):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)

        cs = []
        for c, s in zip(content_features, style_features):
            cs.append(self.multimodal_style_feature_transfer.transfer(c, s).unsqueeze(dim=0))
        cs = torch.cat(cs, dim=0)

        out = self.decoder(cs)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(output_features, content_features)
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        loss = loss_c + gamma * loss_s
        # print('loss: ', loss_c.item(), gamma*loss_s.item())
        return loss
