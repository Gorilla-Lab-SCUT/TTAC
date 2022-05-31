import numpy as np
import torch
import torch.nn as nn
from utils.misc import *


def load_resnet50(net, head, ssh, classifier, args):

    if args.ckpt:
        filename = args.resume + '/ckpt_epoch_{:d}.pth'.format(args.ckpt)
    else:
        filename = args.resume + '/ckpt.pth'
    ckpt = torch.load(filename)
    state_dict = ckpt['model']

    net_dict = {}
    head_dict = {}
    for k, v in state_dict.items():
        if k[:4] == "head":
            k = k.replace("head.", "")
            head_dict[k] = v
        else:
            k = k.replace("encoder.", "ext.")
            k = k.replace("fc.", "head.fc.")
            net_dict[k] = v

    net.load_state_dict(net_dict)
    head.load_state_dict(head_dict)

    print('Loaded model trained jointly on Classification and SimCLR:', filename)


def load_ttt(net, head, ssh, classifier, args, ttt=False):
    if ttt:
        filename = args.resume + '/{}_both_2_15.pth'.format(args.corruption)
    else:
        filename = args.resume + '/{}_both_15.pth'.format(args.corruption)
    ckpt = torch.load(filename)
    net.load_state_dict(ckpt['net'])
    head.load_state_dict(ckpt['head'])
    print('Loaded updated model from', filename)


def corrupt_resnet50(ext, args):
    try:
        # SSL trained encoder
        simclr = torch.load(args.restore + '/simclr.pth')
        state_dict = simclr['model']

        ext_dict = {}
        for k, v in state_dict.items():
            if k[:7] == "encoder":
                k = k.replace("encoder.", "")
                ext_dict[k] = v
        ext.load_state_dict(ext_dict)

        print('Corrupted encoder trained by SimCLR')

    except:
        # Jointly trained encoder
        filename = args.resume + '/ckpt_epoch_{}.pth'.format(args.restore)

        ckpt = torch.load(filename)
        state_dict = ckpt['model']

        ext_dict = {}
        for k, v in state_dict.items():
            if k[:7] == "encoder":
                k = k.replace("encoder.", "")
                ext_dict[k] = v
        # import pdb; pdb.set_trace()
        # print_params(ext)
        ext.load_state_dict(ext_dict)
        print('Corrupted encoder jontly trained on Classification and SimCLR')


def build_resnet50(args):
    from models.BigResNet import SupConResNet, LinearClassifier
    from models.SSHead import ExtractorHead

    print('Building ResNet50...')
    if args.dataset == 'cifar10':
        classes = 10
    elif args.dataset == 'cifar7':
        if not hasattr(args, 'modified') or args.modified:
            classes = 7
        else:
            classes = 10
    elif args.dataset == "cifar100":
        classes = 100

    classifier = LinearClassifier(num_classes=classes).cuda()
    ssh = SupConResNet().cuda()
    head = ssh.head
    ext = ssh.encoder
    net = ExtractorHead(ext, classifier).cuda()
    return net, ext, head, ssh, classifier


def build_model(args):
    from models.ResNet import ResNetCifar as ResNet
    from models.SSHead import ExtractorHead
    print('Building model...')
    if args.dataset == 'cifar10':
        classes = 10
    elif args.dataset == 'cifar7':
        if not hasattr(args, 'modified') or args.modified:
            classes = 7
        else:
            classes = 10
    elif args.dataset == "cifar100":
        classes = 100

    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)
        norm_layer = gn_helper

    if hasattr(args, 'detach') and args.detach:
        detach = args.shared
    else:
        detach = None
    net = ResNet(args.depth, args.width, channels=3, classes=classes, norm_layer=norm_layer, detach=detach).cuda()
    if args.shared == 'none':
        args.shared = None

    if args.shared == 'layer3' or args.shared is None:
        from models.SSHead import extractor_from_layer3
        ext = extractor_from_layer3(net)
        if not hasattr(args, 'ssl') or args.ssl == 'rotation':
            head = nn.Linear(64 * args.width, 4)
        elif args.ssl == 'contrastive':
            head = nn.Sequential(
                nn.Linear(64 * args.width, 64 * args.width),
                nn.ReLU(inplace=True),
                nn.Linear(64 * args.width, 16 * args.width)
            )
        else:
            raise NotImplementedError
    elif args.shared == 'layer2':
        from models.SSHead import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net, args.width, 4)
    ssh = ExtractorHead(ext, head).cuda()

    if hasattr(args, 'parallel') and args.parallel:
        net = torch.nn.DataParallel(net)
        ssh = torch.nn.DataParallel(ssh)
    return net, ext, head, ssh


def test(dataloader, model, **kwargs):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if type(inputs) == list:
            inputs = inputs[0]
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs, **kwargs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    model.train()
    return 1-correct.mean(), correct, losses


def pair_buckets(o1, o2):
    crr = np.logical_and( o1, o2 )
    crw = np.logical_and( o1, np.logical_not(o2) )
    cwr = np.logical_and( np.logical_not(o1), o2 )
    cww = np.logical_and( np.logical_not(o1), np.logical_not(o2) )
    return crr, crw, cwr, cww


def count_each(tuple):
    return [item.sum() for item in tuple]


def plot_epochs(all_err_cls, all_err_ssh, fname, use_agg=True):
    import matplotlib.pyplot as plt
    if use_agg:
        plt.switch_backend('agg')

    plt.plot(np.asarray(all_err_cls)*100, color='r', label='classifier')
    plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised')
    plt.xlabel('epoch')
    plt.ylabel('test error (%)')
    plt.legend()
    plt.savefig(fname)
    plt.close()


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

