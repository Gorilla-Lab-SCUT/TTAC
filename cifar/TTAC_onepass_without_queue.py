import argparse

import torch
import torch.optim as optim
import torch.utils.data as data

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

# ----------------------------------
import copy
import time
import random
import numpy as np

from offline import *
import time
# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default="./data")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_sample', default=1000000, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--outf', default='.')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--ssl', default='contrastive', help='self-supervised task')
parser.add_argument('--without_global', action='store_true', default=False)
parser.add_argument('--without_mixture', action='store_true', default=False)
parser.add_argument('--pro_threshold', default=0.9, type=float)
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

print(args)

my_makedir(args.outf)


class_num = 10 if args.dataset == 'cifar10' else 100

net, ext, head, ssh, classifier = build_resnet50(args)

teset, _ = prepare_test_data(args)
teloader = data.DataLoader(teset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, pin_memory=True, drop_last=False)

# -------------------------------
print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)

# ----------- Offline Feature Summarization ------------
args_align = copy.deepcopy(args)
args_align.ssl = None
_, offlineloader = prepare_train_data(args_align)
ext_src_mu, ext_src_cov, ssh_src_mu, ssh_src_cov, mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh = offline(offlineloader, ext, classifier, head, class_num)

bias = cov_src_ext.max().item() / 30.
template_ext_cov = torch.eye(2048).cuda() * bias

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# # ----------- Test ------------
print('Running...')
print('Error (%)\t\ttest')
err_cls = test(teloader, net)[0]
print(('Epoch %d:' %(0)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))

# ----------- Improved Test-time Training ------------

ext_src_mu = torch.stack(ext_src_mu)
ext_src_cov = torch.stack(ext_src_cov)

ema_ext_mu = ext_src_mu.clone()
ema_ext_cov = ext_src_cov.clone()
ema_ext_total_mu = torch.zeros(2048).float()
ema_ext_total_cov = torch.zeros(2048, 2048).float()

ema_n = torch.zeros(class_num).cuda()
ema_total_n = 0.

if class_num == 10:
    loss_scale = 0.05
    ema_length = 128
else:
    loss_scale = 0.5
    ema_length = 64

correct = []
cumulative_error = []

for te_batch_idx, (te_inputs, te_labels) in enumerate(teloader):
    
    classifier.eval()
    ext.train()

    start = time.time()
    
    optimizer.zero_grad()

    loss = 0.
    
    inputs = te_inputs[0].cuda()

    feat_ext = ext(inputs)
    logit = classifier(feat_ext)

    softmax_logit = logit.softmax(dim=-1)
    pro, pseudo_label = softmax_logit.max(dim=-1)
    pseudo_label_mask = (pro > args.pro_threshold)
    feat_ext2 = feat_ext[pseudo_label_mask]
    pseudo_label2 = pseudo_label[pseudo_label_mask].cuda()

    if not args.without_mixture:
        # Mixture Gaussian
        b, d = feat_ext2.shape
        feat_ext2_categories = torch.zeros(class_num, b, d).cuda() # K, N, D
        feat_ext2_categories.scatter_add_(dim=0, index=pseudo_label2[None, :, None].expand(-1, -1, d), src=feat_ext2[None, :, :])

        num_categories = torch.zeros(class_num, b, dtype=torch.int).cuda() # K, N
        num_categories.scatter_add_(dim=0, index=pseudo_label2[None, :], src=torch.ones_like(pseudo_label2[None, :], dtype=torch.int))

        ema_n += num_categories.sum(dim=1) # K
        alpha = torch.where(ema_n > ema_length, torch.ones(class_num, dtype=torch.float).cuda() / ema_length, 1. / (ema_n + 1e-10))

        delta_pre = (feat_ext2_categories - ema_ext_mu[:, None, :]) * num_categories[:, :, None] # K, N, D
        delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
        new_component_mean = ema_ext_mu + delta
        new_component_cov = ema_ext_cov \
                            + alpha[:, None, None] * ((delta_pre.permute(0, 2, 1) @ delta_pre) - num_categories.sum(dim=1)[:, None, None] * ema_ext_cov) \
                            - delta[:, :, None] @ delta[:, None, :]

        with torch.no_grad():
            ema_ext_mu = new_component_mean.detach()
            ema_ext_cov = new_component_cov.detach()
        
        for label in pseudo_label2.unique():
            if ema_n[label] >= 16:
                source_domain = torch.distributions.MultivariateNormal(ext_src_mu[label, :], ext_src_cov[label, :, :] + template_ext_cov)
                target_domain = torch.distributions.MultivariateNormal(new_component_mean[label, :], new_component_cov[label, :, :] + template_ext_cov)
                loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale / class_num
        
    if not args.without_global:
        # Global Gaussian
        b = feat_ext.shape[0]
        ema_total_n += b
        alpha = 1. / 1280 if ema_total_n > 1280 else 1. / ema_total_n
        delta_pre = (feat_ext - ema_ext_total_mu.cuda())
        delta = alpha * delta_pre.sum(dim=0)
        tmp_mu = ema_ext_total_mu.cuda() + delta
        tmp_cov = ema_ext_total_cov.cuda() + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_total_cov.cuda()) - delta[:, None] @ delta[None, :]
        with torch.no_grad():
            ema_ext_total_mu = tmp_mu.detach().cpu()
            ema_ext_total_cov = tmp_cov.detach().cpu()

        source_domain = torch.distributions.MultivariateNormal(mu_src_ext, cov_src_ext + template_ext_cov)
        target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
        loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #### Test ####
    net.eval()
    with torch.no_grad():
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        correct.append(predicted.cpu().eq(te_labels))
    print('real time error:', 1 - torch.cat(correct).numpy().mean())
    cumulative_error.append(1 - torch.cat(correct).numpy().mean())

print(args.corruption, 'Test time training result:', 1 - torch.cat(correct).numpy().mean())
