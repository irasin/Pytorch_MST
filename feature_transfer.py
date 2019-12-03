import torch
import numpy as np
from maxflow.fastmin import aexpansion_grid
from sklearn.cluster import KMeans


def data_term(content_feature, cluster_centers):
    c = content_feature.permute(1, 2, 0)
    d = torch.matmul(c, cluster_centers)
    c_norm = torch.norm(c, dim=2, keepdim=True)
    s_norm = torch.norm(cluster_centers, dim=0, keepdim=True)
    norm = torch.matmul(c_norm, s_norm)
    d = 1 - d.div(norm)
    return d


def pairwise_term(cluster_centers, lam):
    _, k = cluster_centers.shape
    v = torch.ones((k, k)) - torch.eye(k)
    v = lam * v.to(cluster_centers.device)
    return v


def labeled_whiten_and_color(f_c, f_s, alpha, label):
    try:
        c, h, w = f_c.shape
        cf = (f_c * label).reshape(c, -1)
        c_mean = torch.mean(cf, 1).reshape(c, 1, 1) * label

        cf = cf.reshape(c, h, w) - c_mean
        cf = cf.reshape(c, -1)
        c_cov = torch.mm(cf, cf.t()).div(torch.sum(label).item() / c - 1)
        c_u, c_e, c_v = torch.svd(c_cov)

        # if necessary, use k-th largest eig-value
        k_c = c
        # for i in range(c):
        #     if c_e[i] < 0.00001:
        #         k_c = i
        #         break
        c_d = c_e[:k_c].pow(-0.5)

        w_step1 = torch.mm(c_v[:, :k_c], torch.diag(c_d))
        w_step2 = torch.mm(w_step1, (c_v[:, :k_c].t()))
        whitened = torch.mm(w_step2, cf)

        sf = f_s.t()
        c, k = sf.shape
        s_mean = torch.mean(sf, 1, keepdim=True)
        sf = sf - s_mean
        s_cov = torch.mm(sf, sf.t()).div(k - 1)
        s_u, s_e, s_v = torch.svd(s_cov)

        # if necessary, use k-th largest eig-value
        k_s = c
        # for i in range(c):
        #     if s_e[i] < 0.00001:
        #         k_s = i
        #         break
        s_d = s_e[:k_s].pow(0.5)

        c_step1 = torch.mm(s_v[:, :k_s], torch.diag(s_d))
        c_step2 = torch.mm(c_step1, s_v[:, :k_s].t())
        colored = torch.mm(c_step2, whitened).reshape(c, h, w)
        s_mean = s_mean.reshape(c, 1, 1) * label
        colored = colored + s_mean
        colored_feature = alpha * colored + (1 - alpha) * (f_c * label)
    except:
        # Need fix
        # RuntimeError: MAGMA gesdd : the updating process of SBDSDC did not converge
        colored_feature = f_c * label

    return colored_feature


class MultimodalStyleTransfer:
    def __init__(self, n_cluster, alpha, device='cpu', lam=0.1, max_cycles=None):
        self.k = n_cluster
        self.k_means_estimator = KMeans(n_cluster)
        if (type(alpha) is int or type(alpha) is float) and 0 <= alpha <= 1:
            self.alpha = [alpha] * n_cluster
        elif type(alpha) is list and len(alpha) == n_cluster:
            self.alpha = alpha
        else:
            raise ValueError('Error for alpha')

        self.device = device
        self.lam = lam
        self.max_cycles = max_cycles

    def style_feature_clustering(self, style_feature):
        C, _, _ = style_feature.shape
        s = style_feature.reshape(C, -1).transpose(0, 1)

        self.k_means_estimator.fit(s.to('cpu'))
        labels = torch.Tensor(self.k_means_estimator.labels_).to(self.device)
        cluster_centers = torch.Tensor(self.k_means_estimator.cluster_centers_).to(self.device).transpose(0, 1)

        s = s.to(self.device)
        clusters = [s[labels == i] for i in range(self.k)]

        return cluster_centers, clusters

    def graph_based_style_matching(self, content_feature, style_feature):
        cluster_centers, s_clusters = self.style_feature_clustering(style_feature)

        D = data_term(content_feature, cluster_centers).to('cpu').numpy().astype(np.double)
        V = pairwise_term(cluster_centers, lam=self.lam).to('cpu').numpy().astype(np.double)
        labels = torch.Tensor(aexpansion_grid(D, V, max_cycles=self.max_cycles)).to(self.device)
        return labels, s_clusters

    def transfer(self, content_feature, style_feature):
        labels, s_clusters = self.graph_based_style_matching(content_feature, style_feature)
        f_cs = torch.zeros_like(content_feature)
        for f_s, a, k in zip(s_clusters, self.alpha, range(self.k)):
            label = (labels == k).unsqueeze(dim=0).expand_as(content_feature)
            if (label > 0).any():
                label = label.to(torch.float)
                f_cs += labeled_whiten_and_color(content_feature, f_s, a, label)

        return f_cs

