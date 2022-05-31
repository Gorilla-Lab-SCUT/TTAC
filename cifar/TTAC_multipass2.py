import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

# ----------------------------------
import copy
import math
import random
import numpy as np
import torch.backends.cudnn as cudnn

from offline import *
from utils.contrastive import *
import time
# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default="./data")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_align', default=512, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_sample', default=1000000, type=int)
parser.add_argument('--bnepoch', default=2, type=int)
parser.add_argument('--nepoch', default=500, type=int)
parser.add_argument('--stopepoch', default=25, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--outf', default='.')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--ssl', default='contrastive', help='self-supervised task')
parser.add_argument('--temperature', default=0.5, type=float)
parser.add_argument('--align_ext', action='store_true')
parser.add_argument('--align_ssh', action='store_true')
parser.add_argument('--fix_ssh', action='store_true')
parser.add_argument('--with_ssl', action='store_true', default=False)
parser.add_argument('--with_shot', action='store_true', default=False)
parser.add_argument('--without_global', action='store_true', default=False)
parser.add_argument('--without_mixture', action='store_true', default=False)
parser.add_argument('--filter', default="ours", choices=['ours', 'posterior', 'none'])
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--seed', default=0, type=int)


args = parser.parse_args()

print(args)

my_makedir(args.outf)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True

# -------------------------------

net, ext, head, ssh, classifier = build_resnet50(args)
_, teloader = prepare_test_data(args)

# -------------------------------

args.batch_size = min(args.batch_size, args.num_sample)
args.batch_size_align = min(args.batch_size_align, args.num_sample)

args_align = copy.deepcopy(args)
args_align.ssl = None
args_align.batch_size = args.batch_size_align

_, tr_dataloader = prepare_train_data(args, args.num_sample)

_, trloader_extra = prepare_test_data(args_align, ttt=True, num_sample=args.num_sample)
trloader_extra_iter = iter(trloader_extra)

# -------------------------------

print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

if torch.cuda.device_count() > 1:
    ext = torch.nn.DataParallel(ext)

# ----------- Test ------------

all_err_cls = []
all_err_ssh = []

print('Running...')

if args.fix_ssh:
    optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer = optim.SGD(ssh.parameters(), lr=args.lr, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

criterion = SupConLoss(temperature=args.temperature).cuda()
# -------------------------------

class_num = 10 if args.dataset == 'cifar10' else 100

# ----------- Offline Feature Summarization ------------
_, offlineloader = prepare_train_data(args_align)

ext_src_mu, ext_src_cov, ssh_src_mu, ssh_src_cov, mu_src_ext, cov_src_ext, mu_src_ssh, cov_src_ssh = offline(offlineloader, ext, classifier, head, class_num)
bias = cov_src_ext.max().item() / 30.
bias2 = cov_src_ssh.max().item() / 30.
template_ext_cov = torch.eye(2048).cuda() * bias
template_ssh_cov = torch.eye(128).cuda() * bias2

print('Error (%)\t\ttest')
err_cls = test(teloader, net)[0]
print(('Epoch %d:' %(0)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))


# ----------- Improved Test-time Training ------------

ext_src_mu = torch.stack(ext_src_mu)
ext_src_cov = torch.stack(ext_src_cov) + template_ext_cov[None, :, :]

source_component_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)
target_compoent_distribution = torch.distributions.MultivariateNormal(ext_src_mu, ext_src_cov)

is_both_activated=False
sample_predict_ema_logit = torch.zeros(len(tr_dataloader.dataset), class_num, dtype=torch.float)
sample_predict_alpha = torch.ones(len(tr_dataloader.dataset), dtype=torch.float)
ema_alpha = 0.9

ema_n = torch.zeros(class_num).cuda()
ema_ext_mu = ext_src_mu.clone()
ema_ext_cov = ext_src_cov.clone()

ema_ext_total_mu = torch.zeros(2048).float()
ema_ext_total_cov = torch.zeros(2048, 2048).float()

ema_ssh_total_mu = torch.zeros(128).float()
ema_ssh_total_cov = torch.zeros(128, 128).float()


ema_total_n = 0.
if class_num == 10:
    ema_length = 128
    epoch_bias = 0
else: 
    ema_length = 64
    epoch_bias = 10

if class_num == 10:
    loss_scale = 0.05
else:
    loss_scale = 0.5

for epoch in range(1, args.nepoch+1):
    tic = time.time()

    if args.fix_ssh:
        classifier.eval()
        head.eval()
    else:
        classifier.train()
        head.train()
    ext.train()

    sample_predict_alpha = torch.where(sample_predict_alpha < 1, sample_predict_alpha + 0.2, torch.ones_like(sample_predict_alpha))

    for batch_idx, (inputs, labels) in enumerate(tr_dataloader):
        optimizer.zero_grad()

        if args.with_ssl:
            images = torch.cat([inputs[0], inputs[1]], dim=0)
            images = images.cuda(non_blocking=True)
            indexes = inputs[-1]
            bsz = labels.shape[0]
            backbone_features = ext(images)
            features = F.normalize(head(backbone_features), dim=1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features)
            loss.backward()
            del loss

        if is_both_activated:
            loss = 0.
            try:
                inputs, labels = next(trloader_extra_iter)
            except StopIteration:
                del trloader_extra_iter
                trloader_extra_iter = iter(trloader_extra)
                inputs, labels = next(trloader_extra_iter)

            inputs, indexes = inputs
            inputs = inputs.cuda()

            feat_ext = ext(inputs)
            logit = classifier(feat_ext)
            feat_ssh = head(feat_ext)

            with torch.no_grad():
                ext.eval()
                origin_images = inputs
                origin_image_index = indexes
                predict_logit = net(origin_images)
                softmax_logit = predict_logit.softmax(dim=1).cpu()

                old_logit = sample_predict_ema_logit[origin_image_index, :]
                max_val, max_pos = softmax_logit.max(dim=1)
                old_max_val = old_logit[torch.arange(max_pos.shape[0]), max_pos]
                accept_mask = max_val > (old_max_val - 0.001)

                sample_predict_alpha[origin_image_index] = torch.where(accept_mask, sample_predict_alpha[origin_image_index], torch.zeros_like(accept_mask).float())

                sample_predict_ema_logit[origin_image_index, :] = \
                    torch.where(sample_predict_ema_logit[origin_image_index, :] == torch.zeros(class_num), \
                                softmax_logit, \
                                (1 - ema_alpha) * sample_predict_ema_logit[origin_image_index, :] + ema_alpha * softmax_logit)
                
                pro, pseudo_label = sample_predict_ema_logit[origin_image_index].max(dim=1)
                ext.train()
                del predict_logit

            if args.filter == 'ours':
                pseudo_label_mask = (sample_predict_alpha[origin_image_index] == 1) & (pro > 0.9)
                feat_ext2 = feat_ext[pseudo_label_mask]
                feat_ssh2 = feat_ssh[pseudo_label_mask]
                pseudo_label2 = pseudo_label[pseudo_label_mask].cuda()
            elif args.filter == 'none':
                feat_ext2 = feat_ext
                feat_ssh2 = feat_ssh
                pseudo_label2 = pseudo_label.cuda()
            elif args.filter == 'posterior':
                with torch.no_grad():
                    posterior = target_compoent_distribution.log_prob(feat_ext[:, None, :]) # log prob
                    posterior_tmp = posterior.max(dim=1, keepdim=True)[0] - math.log((2 ** 127) / 10) # B, K
                    posterior -= posterior_tmp
                    posterior = posterior.exp() # prob / exp(posterior_tmp)
                    posterior /= posterior.sum(dim=1, keepdim=True)
                    posterior = posterior.transpose(0, 1).detach()  # K, N
            else:
                raise Exception("%s filter type has not yet been implemented." % args.filter)


            if args.align_ext:
                if not args.without_mixture:
                    # Mixture Gaussian
                    if args.filter != 'posterior':
                        for label in pseudo_label2.unique():
                            feat_ext_per_category = feat_ext2[pseudo_label2 == label, :]

                            b = feat_ext_per_category.shape[0]
                            ema_n[label] += b
                            alpha = 1. / ema_length if ema_n[label] > ema_length else 1. / ema_n[label]

                            ema_ext_mu_that = ema_ext_mu[label, :]
                            ema_ext_cov_that = ema_ext_cov[label, :, :]
                            delta_pre = feat_ext_per_category - ema_ext_mu_that

                            delta = alpha * delta_pre.sum(dim=0)
                            tmp_mu = ema_ext_mu_that + delta
                            tmp_cov = ema_ext_cov_that + alpha * (delta_pre.t() @ delta_pre - b * ema_ext_cov_that) - delta[:, None] @ delta[None, :]

                            with torch.no_grad():
                                ema_ext_mu[label, :] = tmp_mu.detach()
                                ema_ext_cov[label, :, :] = tmp_cov.detach()

                            if epoch > epoch_bias:
                                source_domain = torch.distributions.MultivariateNormal(ext_src_mu[label, :], ext_src_cov[label, :, :] + template_ext_cov)
                                target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ext_cov)
                                loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale / class_num
                    else:
                        feat_ext2_categories = feat_ext[None, :, :].expand(class_num, -1, -1) # K, N, D
                        num_categories = posterior # K, N
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
                        
                        if epoch > epoch_bias:
                            target_compoent_distribution.loc = new_component_mean
                            target_compoent_distribution.covariance_matrix = new_component_cov + template_ext_cov
                            target_compoent_distribution._unbroadcasted_scale_tril = torch.linalg.cholesky(new_component_cov + template_ext_cov)
                            loss += (torch.distributions.kl_divergence(source_component_distribution, target_compoent_distribution) \
                                    + torch.distributions.kl_divergence(target_compoent_distribution, source_component_distribution)).mean() * loss_scale

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

                if args.without_mixture and args.without_global:
                    logit2 = logit[pseudo_label_mask.cuda()]
                    loss += F.cross_entropy(logit2, pseudo_label2) * loss_scale * 2

            if args.align_ssh:  
                b = feat_ssh.shape[0]
                alpha = 1. / 1280 if ema_total_n > 1280 else 1. / ema_total_n
                delta_pre = (feat_ssh - ema_ssh_total_mu.cuda())
                delta = alpha * delta_pre.sum(dim=0)
                tmp_mu = ema_ssh_total_mu.cuda() + delta
                tmp_cov = ema_ssh_total_cov.cuda() + alpha * (delta_pre.t() @ delta_pre - b * ema_ssh_total_cov.cuda()) - delta[:, None] @ delta[None, :]

                with torch.no_grad():
                    ema_ssh_total_mu = tmp_mu.detach().cpu()
                    ema_ssh_total_cov = tmp_cov.detach().cpu()
                source_domain = torch.distributions.MultivariateNormal(mu_src_ssh, cov_src_ssh + template_ssh_cov)
                target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + template_ssh_cov)
                loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * loss_scale
            
                
            if args.with_shot:
                ent_loss = softmax_entropy(logit).mean(0)
                softmax_out = F.softmax(logit, dim=-1)
                msoftmax = softmax_out.mean(dim=0)
                ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
                loss += ent_loss * loss_scale * 2

            try:
                loss.backward()
            except:
                pass
            finally:
                del loss

        if epoch > args.bnepoch:
            optimizer.step()
            optimizer.zero_grad()

    err_cls = test(teloader, net)[0]
    all_err_cls.append(err_cls)

    toc = time.time()
    print(('Epoch %d/%d (%.0fs):' %(epoch, args.nepoch, toc-tic)).ljust(24) +
                    '%.2f\t\t' %(err_cls*100))
    
    # both components
    if not is_both_activated and epoch > args.bnepoch:
        is_both_activated = True

    # termination
    if epoch > (args.stopepoch + 1) and all_err_cls[-args.stopepoch] < min(all_err_cls[-args.stopepoch+1:]):
        print("{} Termination: {:.2f}".format(args.corruption, all_err_cls[-args.stopepoch]*100))
        break

    # save
    if epoch > args.bnepoch and len(all_err_cls) > 2 and all_err_cls[-1] < min(all_err_cls[:-1]):
        state = {'net': net.state_dict(), 'head': head.state_dict()}
        save_file = os.path.join(args.outf, args.corruption + '.pth')
        torch.save(state, save_file)
        print('Save model to', save_file)

    # lr decay
    scheduler.step(err_cls)

