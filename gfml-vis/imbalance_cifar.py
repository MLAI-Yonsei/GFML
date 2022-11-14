import numpy as np
import torch
import torchvision
import os


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    '''
    - Imbalance CIFAR10을 위한 class입니다.
    - imb_type과 imb_factor 가지고 class별 데이터 개수 조절 가능합니다.
    (위 argument에 따른 class별 데이터 개수는 get_img_num_per_cls 함수를 살펴보시면 좋을 것 같습니다! 아니면 객체 생성해놓고 get_cls_num_list로도 파악 가능합니다)
    - loader 실행 시 각 class별 데이터 개수와 imb_factor print하도록 만들어놨습니다.
    imbalance_cifar.py from "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" Cao et al.
    '''
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
      
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    '''
    Imbalance CIFAR10 class 상속해서 만든 class로 방식은 똑같습니다.
    '''
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d'],]
    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    '''
    class of Sampler for Random Over Sampling 
    (OverSampling 안 하실꺼면 지우셔도 될 것 같습니다!)
    '''
    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

#########################################################
####### loader 적용 예시
import torchvision.transforms as transforms
transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
transform_test = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def loaders(
    dataset='IMBALANCECIFAR10',
    batch_size=128,
    num_workers=4,
    transform_train=transform_train,
    transform_test=transform_test,
    use_validation=True,
    val_size=5000,
    shuffle_train=True,
    imb_factor=0.01,
    ROS=False,
    **kwargs
):
    '''
    따로 validation set 만드는 코드는 없어서 (train / test split만 된 상황) 필요하신 경우
    use_validation, val_size 인자 이용해서 추가하시면 될 것 같습니다.
    '''
    # Imbalance CIFAR10
    if dataset == "IMBALANCECIFAR10":
        train_set = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        num_classes = max(train_set.targets) + 1
        print(f'Dataset is Imbalance CIFAR10 (imb_factor={imb_factor})')
        nSamples = train_set.get_cls_num_list()
        print(f"nTrainSamples per Class : {nSamples}")

        if ROS:
            print('Using Oversampling to Address Class Imbalance')
            train_sampler = ImbalancedDatasetSampler(train_set)
            return (
            {
                "train": torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=train_sampler
                ),
                "test": torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                ),
            },
            num_classes,
            )

        return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, #100
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
        )
################################################################################

############################ Imbalance CIFAR100 ################################
    if dataset == "IMBALANCECIFAR100":
        train_set = IMBALANCECIFAR100(root='./data', imb_type='exp', imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

        num_classes = max(train_set.targets) + 1
        print(f'Dataset is Imbalance CIFAR100 (imb_factor={imb_factor})')
        nSamples = train_set.get_cls_num_list()
        print(f"nTrainSamples per Class : {nSamples}")

        if ROS:
            print('Using Oversampling to Address Class Imbalance ')
            train_sampler = ImbalancedDatasetSampler(train_set)

            return (
              {
                "train": torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=train_sampler
                ),
                "test": torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size, #100
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                ),
              },
              num_classes,
            )

        return (
          {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
          },
          num_classes,
        )
################################################################################