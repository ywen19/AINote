import os
import struct
import numpy as np


class Dataset(object):

    def __init__(self, data_root, mode='train', num_classes=10):
        assert mode in ['train', 'val', 'test']
        
        self.num_classes = 10

        # load images and labels
        kind = {'train': 'train', 'val': 'train', 'test': 't10k'}[mode]
        labels_path = os.path.join(data_root, '{}-labels.idx1-ubyte'.format(kind))
        images_path = os.path.join(data_root, '{}-images.idx3-ubyte'.format(kind))

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        if mode == 'train':
            # training images and labels
            self.images = images[:55000]  # shape: (55000, 784)
            self.labels = labels[:55000]  # shape: (55000,)

        elif mode == 'val':
            # validation images and labels
            self.images = images[55000:]  # shape: (5000, 784)
            self.labels = labels[55000:]  # shape: (5000, )

        else:
            # test data
            self.images = images  # shape: (10000, 784)
            self.labels = labels  # shape: (10000, )


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Normalize from [0, 255.] to [0., 1.0], and then subtract by the mean value
        image = image / 255.0
        image = image - np.mean(image)

        return image, label


class IterationBatchSampler(object):

    def __init__(self, dataset, max_epoch, batch_size=2, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def prepare_epoch_indices(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        num_iteration = len(indices) // self.batch_size + int(len(indices) % self.batch_size)
        assert len(indices)%num_iteration==0.0, "Split requires a equal division. Change batch size! "
        self.batch_indices = np.split(indices, num_iteration)

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)


class Dataloader(object):

    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        self.sampler.prepare_epoch_indices()

        for batch_indices in self.sampler:
            batch_images = []
            batch_labels = []
            for idx in batch_indices:
                img, label = self.dataset[idx]
                batch_images.append(img)
                batch_labels.append(label)

            batch_images = np.stack(batch_images)
            batch_labels = np.stack(batch_labels)

            yield batch_images, batch_labels

    def __len__(self):
        return len(self.sampler)


def build_dataloader(data_root, max_epoch, batch_size, shuffle=False, mode='train'):
    dataset = Dataset(data_root, mode)
    sampler = IterationBatchSampler(dataset, max_epoch, batch_size, shuffle)
    data_lodaer = Dataloader(dataset, sampler)
    return data_lodaer
