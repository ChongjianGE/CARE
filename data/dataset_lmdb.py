from torchvision.datasets import ImageFolder
from data.folder2lmdb import ImageFolderLMDB


class SSL_Dataset(ImageFolderLMDB):
    def __init__(self, transform=None,
                 root="/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/train.lmdb", target_transform=None, is_valid_file=None):
        super(SSL_Dataset, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )
        # self.image_dir + f"imagenet_{'train' if stage in ('train', 'ft') else 'val'}"

        if transform is not None:
            if isinstance(transform, list) and len(transform) > 1:
                self.transform, self.transform_k = transform
            else:
                self.transform, self.transform_k = transform, None
        else:
            raise ValueError("Transform function missing!")

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        img1 = self.transform(sample)
        if self.transform_k is not None:
            img2 = self.transform_k(sample)
        else:
            img2 = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target


class ImageNet(ImageFolderLMDB):
    def __init__(self, train, transform=None):
        root = "/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/{}.lmdb".format('train' if train else 'val')
        super(ImageNet, self).__init__(root, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
