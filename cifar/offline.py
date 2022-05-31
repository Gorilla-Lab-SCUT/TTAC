import torch
from discrepancy import *


def offline(trloader, ext, classifier, head, class_num=10):
    ext.eval()
    
    feat_stack = [[] for i in range(class_num)]
    ssh_feat_stack = [[] for i in range(class_num)]

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(trloader):

            feat = ext(inputs.cuda())
            predict_logit = classifier(feat)
            ssh_feat = head(feat)
            
            pseudo_label = predict_logit.max(dim=1)[1]

            for label in pseudo_label.unique():
                label_mask = pseudo_label == label
                feat_stack[label].extend(feat[label_mask, :])
                ssh_feat_stack[label].extend(ssh_feat[label_mask, :])
    ext_mu = []
    ext_cov = []
    ext_all = []

    ssh_mu = []
    ssh_cov = []
    ssh_all = []
    for feat in feat_stack:
        ext_mu.append(torch.stack(feat).mean(dim=0))
        ext_cov.append(covariance(torch.stack(feat)))
        ext_all.extend(feat)
    
    for feat in ssh_feat_stack:
        ssh_mu.append(torch.stack(feat).mean(dim=0))
        ssh_cov.append(covariance(torch.stack(feat)))
        ssh_all.extend(feat)
    
    ext_all = torch.stack(ext_all)
    ext_all_mu = ext_all.mean(dim=0)
    ext_all_cov = covariance(ext_all)

    ssh_all = torch.stack(ssh_all)
    ssh_all_mu = ssh_all.mean(dim=0)
    ssh_all_cov = covariance(ssh_all)
    return ext_mu, ext_cov, ssh_mu, ssh_cov, ext_all_mu, ext_all_cov, ssh_all_mu, ssh_all_cov
