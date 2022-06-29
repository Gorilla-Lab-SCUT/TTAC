import torch
import os

def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / n
    return cov

def offline(trloader, ext, num_classes=1000):
    if os.path.exists('offline.pth'):
        data = torch.load('offline.pth')
        return data

    ext.eval()

    feat_ext_mean = torch.zeros(2048).cuda()
    feat_ext_variance = torch.zeros(2048, 2048).cuda()

    feat_ext_mean_categories = torch.zeros(num_classes, 2048).cuda() # K, D
    feat_ext_variance_categories = torch.zeros(num_classes, 2048).cuda()

    ema_n = torch.zeros(num_classes).cuda()
    ema_total_n = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(trloader):
            feat = ext(inputs[0].cuda()) # N, D
            b, d = feat.shape
            labels = labels.cuda()

            feat_ext_categories = torch.zeros(num_classes, b, d).cuda()
            feat_ext_categories.scatter_add_(dim=0, index=labels[None, :, None].expand(-1, -1, d), src=feat[None, :, :])
            
            num_categories = torch.zeros(num_classes, b, dtype=torch.int).cuda()
            num_categories.scatter_add_(dim=0, index=labels[None, :], src=torch.ones_like(labels[None, :], dtype=torch.int))
            ema_n += num_categories.sum(dim=1)
            alpha_categories = 1 / (ema_n + 1e-10)  # K
            delta_pre = (feat_ext_categories - feat_ext_mean_categories[:, None, :]) * num_categories[:, :, None] # K, N, D
            delta = alpha_categories[:, None] * delta_pre.sum(dim=1) # K, D
            feat_ext_mean_categories += delta
            feat_ext_variance_categories += alpha_categories[:, None] * ((delta_pre ** 2).sum(dim=1) - num_categories.sum(dim=1)[:, None] * feat_ext_variance_categories) \
                                          - delta ** 2
            
            ema_total_n += b
            alpha = 1 / (ema_total_n + 1e-10)
            delta_pre = feat - feat_ext_mean[None, :] # b, d
            delta = alpha * (delta_pre).sum(dim=0)
            feat_ext_mean += delta
            feat_ext_variance += alpha * (delta_pre.t() @ delta_pre - b * feat_ext_variance) - delta[:, None] @ delta[None, :]
            print('offline process rate: %.2f%%\r' % ((batch_idx + 1) / len(trloader) * 100.), end='')


    torch.save((feat_ext_mean, feat_ext_variance, feat_ext_mean_categories, feat_ext_variance_categories), 'offline.pth')
    return feat_ext_mean, feat_ext_variance, feat_ext_mean_categories, feat_ext_variance_categories