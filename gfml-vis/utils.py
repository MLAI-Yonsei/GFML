import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy, SVHNPolicy
from criterions import LabelSmoothingCrossEntropyLoss
from da import RandomCropPaste

from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'vit_gra':
        from vit import ViT
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            qk_mode=args.model_name
            )

    elif args.model_name =='resnet18':
        print("ResNet-18")
        from torchvision.models import resnet18
        net = resnet18(num_classes=args.num_classes)
        # num_ftrs = net.fc.in_features
        # net.fc = nn.Linear(num_ftrs, args.num_classes)

    elif args.model_name == 'densenet121':
        print("DenseNet121")
        from torchvision.models import densenet121

        net = densenet121(num_classes=args.num_classes)
        # num_ftrs = net.classifier.in_features
        # net.classifier = nn.Linear(num_ftrs, args.num_classes)

    elif args.model_name == 'efficientnetb2':
        print("EfficientNet_B2")
        from torchvision.models import efficientnet_b2
        net = efficientnet_b2(num_classes=args.num_classes)

    elif args.model_name == 'vgg19':
        print('VGG-19')
        from torchvision.models import vgg19
        net = vgg19(num_classes=args.num_classes)
        # num_ftrs = net.classifier[6].in_features
        # net.classifier[6] = nn.Linear(4096, args.num_classes)
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == 'cimb10':
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = IMBALANCECIFAR10(root=root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True, download=True, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    elif args.dataset == 'cimb100':
        args.in_c = 3
        args.num_classes = 100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = IMBALANCECIFAR100(root=root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                                    download=True, transform=train_transform)
        test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)


    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    if args.model_name == "vit_gra":
        experiment_name+=f"_mass_{args.mass_pos}"
        experiment_name += f"_lamb_{args.lamb_gra}_x"
        experiment_name+=f"_dist_{args.dist_mode}"
        if args.mix_on:
            experiment_name+= f"_mix_{args.mix_mode}_on"
            if args.mix_mode == "split":
                experiment_name+=f"_split_{args.num_split}"
                if args.split_mix == 1:
                    experiment_name+=f"split_mix_{args.split_mix}"
            experiment_name+=f"_mix_{args.lamb_mix}"
        if args.score_norm:
            experiment_name+="_score_norm"
    print(f"Experiment:{experiment_name}")
    return experiment_name
