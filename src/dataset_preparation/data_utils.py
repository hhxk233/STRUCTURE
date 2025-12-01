import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.core.src.datasets.downstream_tasks.coco_dataset import CocoCaptionDataset
from src.core.src.datasets.downstream_tasks.flickr30k_dataset import Flickr30kDataset

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class CombinedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_files: List[Union[str, Path]],
        feature_name: str,
    ):
        self.feature_files = feature_files
        self.n_samples = []
        self.features = {}
        for feat_file in self.feature_files:
            feat_file = Path(feat_file)
            saved_artifacts = torch.load(feat_file, weights_only=False)
            self.features[feat_file.name] = saved_artifacts[feature_name]
            self.n_samples.append(len(self.features[feat_file.name]))
            logger.info(f"Loaded eval features from: {feat_file}")

        # make sure that all features have the same length
        assert self.n_samples.count(self.n_samples[0]) == len(self.n_samples)

    def __len__(self):
        return self.n_samples[0]

    def __getitem__(self, idx):
        item = None
        for feat in self.features.values():
            if item is None:
                item = (feat[idx],)
            else:
                item += (feat[idx],)
        return item


class FeatureDataset(Dataset):
    def __init__(
        self,
        feature_file: Union[str, Path],
        feature_name: str,
        target_name: Optional[str] = None,
    ):
        self.feature_file = feature_file
        saved_artifacts = torch.load(feature_file, weights_only=False)
        self.features = saved_artifacts[feature_name]
        self.targets = None
        if target_name is not None:
            self.targets = saved_artifacts[target_name]
            assert self.features.shape[0] == self.targets.shape[0]
        del saved_artifacts
        logger.info(f"Loaded eval features from: {feature_file}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], int(self.targets[idx])
        return self.features[idx]


def get_meta_dict(dataset):
    meta_dict = {}

    # If dataset contains sub-datasets, collect attributes from them.
    if hasattr(dataset, "datasets"):

        def gather_attr(attr_name):
            return [
                getattr(d, attr_name) for d in dataset.datasets if hasattr(d, attr_name)
            ]

        image_files = gather_attr("_image_files")
        labels = gather_attr("_labels")
        # Combine both '_samples' and 'samples' from sub-datasets.
        samples = gather_attr("_samples") + gather_attr("samples")
        # Use the first available class_to_idx from sub-datasets.
        class_to_idx = next(
            (d.class_to_idx for d in dataset.datasets if hasattr(d, "class_to_idx")),
            None,
        )

        if image_files:
            meta_dict["image_files"] = image_files
        if labels:
            meta_dict["labels"] = labels
        if samples:
            meta_dict["samples"] = samples
        if class_to_idx is not None:
            meta_dict["class_to_idx"] = class_to_idx

    # Override or add with the main dataset attributes if they exist.
    for attr, key in (
        ("_image_files", "image_files"),
        ("_labels", "labels"),
        ("class_to_idx", "class_to_idx"),
        ("_samples", "samples"),
        ("samples", "samples"),
    ):
        if hasattr(dataset, attr):
            meta_dict[key] = getattr(dataset, attr)

    return meta_dict


class MyFER2013(dsets.FER2013):
    _RESOURCES = {
        "train": ("train_split.csv", "aa1bdf3e64bc6697783ce586283a2b74"),
        "test": ("test_split.csv", "8576e0f5a806d7b337d6eeda66d71dc0"),
    }


class MySUN397(dsets.SUN397):
    def __init__(
        self, root, transform, target_transform=None, partition=1, split="train"
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        self.partition = partition
        self.split = split
        self.filter()

    def filter(self):
        split_str = (
            f"Training_{self.partition:02d}.txt"
            if self.split == "train"
            else f"Testing_{self.partition:02d}.txt"
        )
        with open(self._data_dir / split_str) as f:
            self._image_files = f.read().splitlines()
            self._image_files = [
                self._data_dir / elem[1:] for elem in self._image_files
            ]

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])]
            for path in self._image_files
        ]


def _ensure_rgb_image(image: torch.Tensor):
    if torch.is_tensor(image):
        if image.shape[0] != 3:
            image = image.expand(3, -1)
        return image
    else:
        return image.convert("RGB")


def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")


def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return transforms.ToTensor()(x)


def get_default_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            _safe_to_tensor,
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def get_dataloaders(dataset, transform, batch_size, root_dir="data"):
    if transform is None:
        # just dummy resize -> both CLIP and DINO support 224 size of the image
        transform = get_default_transforms()
    train_dataset, val_dataset = get_datasets(dataset, transform, root_dir)
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    valloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=10
    )
    return trainloader, valloader


def get_datasets(dataset, transform, root_dir: Union[str, Path] = "./data", **kwargs):
    data_path = root_dir

    if dataset == "food101":
        train_dataset = dsets.Food101(
            root=data_path, split="train", transform=transform, download=True
        )
        val_dataset = dsets.Food101(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "flickr30":
        flickr_path = os.path.join(data_path, "flickr30k_images")
        train_dataset = Flickr30kDataset(
            root_dir=flickr_path,
            meta_path=os.path.join(flickr_path, "results.csv"),
            split=["train", "val"],
            transform=transform,
            **kwargs,
        )
        val_dataset = Flickr30kDataset(
            root_dir=flickr_path,
            meta_path=os.path.join(flickr_path, "results.csv"),
            split="test",
            transform=transform,
            **kwargs,
        )

    elif dataset == "cifar10":
        train_dataset = dsets.CIFAR10(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.CIFAR10(
            root=data_path, train=False, transform=transform, download=True
        )

    elif dataset == "cifar100":
        train_dataset = dsets.CIFAR100(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.CIFAR100(
            root=data_path, train=False, transform=transform, download=True
        )

    elif dataset == "birdsnap":
        """
        Manually download wget https://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz
        and run python get_birdsnap.py (Requires python2)
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "birdsnap/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "birdsnap/test"), transform=transform
        )

    elif dataset == "sun397":
        """
        Manually download http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
        and https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
        and move Training_0* / Testing_0* splits to unzipped SUN397 dir
        """
        train_dataset = MySUN397(
            root=data_path, partition=1, split="train", transform=transform
        )
        val_dataset = MySUN397(
            root=data_path, partition=1, split="test", transform=transform
        )

    elif dataset == "cars":
        """
        # might require manual download
        # https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?resource=download&sort=recent-comments
        # https://github.com/pytorch/vision/issues/7545
        # https://github.com/nguyentruonglau/stanford-cars/blob/main/labeldata/cars_test_annos_withlabels.mat

        As noted in https://github.com/pytorch/vision/issues/7545
        1) Download from Kaggle https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?resource=download&sort=recent-comments
            unzip into folder "stanford_cars",
        2) Download https://github.com/pytorch/vision/files/11644847/car_devkit.tgz,
            and unzip into folder "stanford_cars"
        3) Download https://github.com/nguyentruonglau/stanford-cars/blob/main/labeldata/cars_test_annos_withlabels.mat
            to the folder "stanford_cars"
        4) Make sure the content structure is
                -stanford_cars/
                    -cars_train/
                        - xxx.jpg
                    -cars_test/
                        - xxx.jpg
                -devkit/
                    -car_train_annos.mat
                -cars_test_annos_withlabels.mat
        """
        train_dataset = dsets.StanfordCars(
            root=data_path, split="train", transform=transform, download=False
        )
        val_dataset = dsets.StanfordCars(
            root=data_path, split="test", transform=transform, download=False
        )

    elif dataset == "aircraft":
        """
        Run python prepare_aircraft.py -i ./data/datasets/aircraft -o ./data/datasets/aircraft --download
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "aircraft/trainval"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "aircraft/test"), transform=transform
        )

    elif dataset == "dtd":
        tmp_dataset1 = dsets.DTD(
            root=data_path, split="train", transform=transform, download=True
        )
        tmp_dataset2 = dsets.DTD(
            root=data_path, split="val", transform=transform, download=True
        )
        train_dataset = ConcatDataset((tmp_dataset1, tmp_dataset2))
        val_dataset = dsets.DTD(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "pets":
        """
        Run python prepare_pets.py -i ./data/datasets/pets -o ./data/datasets/pets --download
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "pets/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "pets/test"), transform=transform
        )

    elif dataset == "caltech101":
        """
        Download manually from https://data.caltech.edu/records/mzrjq-6wc02
        and unzip to ./data/datasets/caltech-101/101_ObjectCategories
        """
        tmp_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "caltech-101/101_ObjectCategories"),
            transform=transform,
        )
        tmp_targets = np.array(tmp_dataset.targets)
        subset = []
        for t in np.unique(tmp_targets):
            np.random.seed(42)
            subset.extend(
                np.random.choice(np.where(tmp_targets == t)[0], size=30, replace=False)
            )
        subset_val = list(set([i for i in range(len(tmp_targets))]) - set(subset))
        train_dataset = Subset(tmp_dataset, subset)
        val_dataset = Subset(tmp_dataset, subset_val)

    elif dataset == "flowers":
        tmp_dataset1 = dsets.Flowers102(
            root=data_path, split="train", transform=transform, download=True
        )
        tmp_dataset2 = dsets.Flowers102(
            root=data_path, split="val", transform=transform, download=True
        )
        train_dataset = ConcatDataset((tmp_dataset1, tmp_dataset2))
        val_dataset = dsets.Flowers102(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "mnist":
        train_dataset = dsets.MNIST(
            root=data_path, train=True, transform=transform, download=True
        )
        val_dataset = dsets.MNIST(
            root=data_path, train=False, transform=transform, download=True
        )

    elif dataset == "fer2013":
        """
        Download manually from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
        and unzip to ./data/datasets/fer2013
        Split fer2013.csv to train.csv (Usage = Training) and test.csv (Usage = PrivateTest/PublicTest) manually
        Then change md5sum in MyFER2013 class
        """
        train_dataset = dsets.FER2013(
            root=data_path, split="train", transform=transform
        )
        val_dataset = dsets.FER2013(root=data_path, split="test", transform=transform)

    elif dataset == "stl10":
        train_dataset = dsets.STL10(
            root=data_path, split="train", transform=transform, download=True
        )
        val_dataset = dsets.STL10(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "eurosat":
        """
        Manually download wget https://madm.dfki.de/files/sentinel/EuroSAT.zip
        and then unzip to ./data/datasets/eurosat
        """
        tmp_dataset = dsets.EuroSAT(root=data_path, transform=transform, download=False)
        tmp_targets = np.array(tmp_dataset.targets)
        subset_train = []
        subset_val = []
        for t in np.unique(tmp_targets):
            np.random.seed(42)
            subset = np.random.choice(
                np.where(tmp_targets == t)[0], size=1500, replace=False
            )
            subset_train.extend(subset[:1000])
            subset_val.extend(subset[1000:])
        train_dataset = Subset(tmp_dataset, subset_train)
        val_dataset = Subset(tmp_dataset, subset_val)

    elif dataset == "resisc45":
        """
        Manually download https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs and unrar to ./data/datasets/tmp
        Then run prepare_resisc45.py -i ./data/datasets/tmp -o ./data/datasets/resisc45
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "resisc45/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "resisc45/test"), transform=transform
        )

    elif dataset == "gtsrb":
        train_dataset = dsets.GTSRB(
            root=data_path, split="train", transform=transform, download=True
        )
        val_dataset = dsets.GTSRB(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "kitti":
        """
        Manually download and unzip to ./data/Kitti
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
        So you will have structure like ./data/Kitti/training/image_2 and
        ./data/Kitti/training/label_2
        Then run python prepare_kitti.py -i ./data/Kitti -o ./data/Kitti
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "Kitti/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "Kitti/val"), transform=transform
        )

    elif dataset == "country211":
        tmp_dataset1 = dsets.Country211(
            root=data_path, split="train", transform=transform, download=True
        )
        tmp_dataset2 = dsets.Country211(
            root=data_path, split="valid", transform=transform, download=True
        )
        train_dataset = ConcatDataset((tmp_dataset1, tmp_dataset2))
        val_dataset = dsets.Country211(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "pcam":
        tmp_dataset1 = dsets.PCAM(
            root=data_path, split="train", transform=transform, download=True
        )
        tmp_dataset2 = dsets.PCAM(
            root=data_path, split="val", transform=transform, download=True
        )
        train_dataset = ConcatDataset((tmp_dataset1, tmp_dataset2))
        val_dataset = dsets.PCAM(
            root=data_path, split="test", transform=transform, download=True
        )

    elif dataset == "ucf101":
        """
        It requires pip install av pyunpack patool
        Just run python prepare_ucf101.py -i ./data/ucf101 -o ./data/ucf101 --download
        And everything will be setup
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "ucf101/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "ucf101/val"), transform=transform
        )

    elif dataset == "kinetics700":
        """
        It requires pip install av
        Just run python prepare_k700.py -i ./data/datasets/k700 -o ./data/datasets/k700 --download --download_workers 50
        And everything will be setup. Try to maximize --download_workers, otherwise it will take many hours to download everything
        Overall, it might take up to day and a half to setup the dataset
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "k700/train_images"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "k700/val_images"), transform=transform
        )

    elif dataset == "clevr":
        """
        Manually download https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
        and unzip to ./data/datasets/CLEVR_v1.0
        Then run prepare_clevr.py
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "CLEVR_v1.0/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "CLEVR_v1.0/test"), transform=transform
        )

    elif dataset == "hatefulmemes":
        """
        Manually download https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset
        and unzip to ./data/datasets/hatefulmemes
        Then run prepare_memes.py
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "hatefulmemes/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "hatefulmemes/test"), transform=transform
        )

    elif dataset == "sst":
        """
        Manually download and put to ./data/datasets/rendered-sst2
        wget https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz
        tar zxvf rendered-sst2.tgz
        """
        tmp_dataset1 = dsets.ImageFolder(
            root=os.path.join(data_path, "rendered-sst2/train"), transform=transform
        )
        tmp_dataset2 = dsets.ImageFolder(
            root=os.path.join(data_path, "rendered-sst2/valid"), transform=transform
        )
        train_dataset = ConcatDataset((tmp_dataset1, tmp_dataset2))
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "rendered-sst2/test"), transform=transform
        )

    elif dataset == "imagenet":
        """
        Manually download from https://www.image-net.org/ and put to ./data/datasets/imagenet
        """
        train_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "imagenet/train"), transform=transform
        )
        val_dataset = dsets.ImageFolder(
            root=os.path.join(data_path, "imagenet/val"), transform=transform
        )
        # Creation of mapping dictionaries to obtain the image classes
        class_mapping_dict = {}
        class_mapping_dict_number = {}
        mapping_class_to_number = {}
        mapping_number_to_class = {}
        i = 0
        with open(data_path / "imagenet/LOC_synset_mapping.txt", "r") as mapping_file:
            for line in mapping_file:
                class_mapping_dict[line[:9].strip()] = line[9:].strip()
                class_mapping_dict_number[i] = line[9:].strip()
                mapping_class_to_number[line[:9].strip()] = i
                mapping_number_to_class[i] = line[:9].strip()
                i += 1
        for d in [train_dataset, val_dataset]:
            d.class_mapping_dict = class_mapping_dict
            d.class_mapping_dict_number = class_mapping_dict_number
            d.mapping_class_to_number = mapping_class_to_number
            d.mapping_number_to_class = mapping_number_to_class
            d.classes = [d.class_mapping_dict.get(x) for x in d.classes]

    elif dataset == "coco":
        coco_path = Path(data_path) / "COCO"
        anno_path = coco_path / "annotations"

        train_path = coco_path / "train2014"
        train_anno = anno_path / "captions_train2014.json"

        val_path = coco_path / "val2014"
        val_anno = anno_path / "captions_val2014.json"

        train_dataset = CocoCaptionDataset(
            annotation_file=train_anno,
            image_dir=train_path,
            transform=transform,
            **kwargs,
        )
        val_dataset = CocoCaptionDataset(
            annotation_file=val_anno,
            image_dir=val_path,
            transform=transform,
            **kwargs,
        )

    return train_dataset, val_dataset


if __name__ == "__main__":
    all_datasets = [
        "food101",
        "cifar10",
        "cifar100",
        "sun397",
        "cars",
        "aircraft",
        "dtd",
        "pets",
        "caltech101",
        "flowers",
        "mnist",
        "fer2013",
        "stl10",
        "eurosat",
        "resisc45",
        "gtsrb",
        "kitti",
        "country211",
        "pcam",
        "ucf101",
        "kinetics700",
        "clevr",
        "hatefulmemes",
        "sst",
        "imagenet",
    ]

    for dataset in all_datasets:
        try:
            ds_train, ds_val = get_datasets(dataset, get_default_transforms(), "./data")
            print(
                f"Successfully load {dataset} ! training set {len(ds_train)}, test set {len(ds_val)}"
            )
        except Exception as e:
            print(f"Error on {dataset}: {e}")
