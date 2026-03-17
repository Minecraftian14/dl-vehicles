from typing import TypedDict, Callable, Any
from torch.utils.data import Dataset, IterableDataset, DataLoader
from PIL import Image
import os, random
import numpy as np
# noinspection PyProtectedMember
from torch.utils.data._utils.collate import default_collate

class_to_idx = {
    "bus": 0,
    "truck": 1,
    "car": 2,
    "motorcycle": 3,
    "none": 4
}

idx_to_class = {v: k for k, v in class_to_idx.items()}

Box = tuple[float, float, float, float]


class Descriptor(TypedDict):
    label: str
    box: Box
    dir: tuple[str, str, str]


def area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


#

class OIDv6Dataset(Dataset):
    available_categories = ('bench', 'bus', 'car', 'cat', 'dog', 'fire_hydrant', 'motorcycle', 'parking_meter',
                            'person', 'piano', 'stop_sign', 'traffic_light', 'traffic_sign', 'truck')

    @staticmethod
    def load_image(
            specimen: str = 'train',
            label: str = None,
            sample: str = None,
            root: str = '..',
            log: bool = False,
            descriptor=None,
            dtype=np.float32,
    ):
        try:
            if label is None: label = random.choice(OIDv6Dataset.available_categories)
            assert specimen in ('train', 'test', 'validation')
            assert label in OIDv6Dataset.available_categories
            directory = os.path.join(root, 'dataset', 'OIDv6', specimen, label)
            if sample is None: sample = random.choice(os.listdir(directory)).rsplit('.', 1)[0]
            if log: print(f'Loading {specimen}/{label}/{sample}')
            image = Image.open(os.path.join(directory, f'{sample}.jpg'))
            if image.mode != 'RGB': image = image.convert('RGB')
            image = np.array(image, dtype=dtype)
            if descriptor is not None: return descriptor, image
            box = tuple(map(float, open(os.path.join(directory, 'labels', f'{sample}.txt')).read().rsplit(' ')[-4:]))
            return {'label': label, 'box': box, 'dir': (specimen, label, sample)}, image
        except Exception as e:
            print(f'Error loading {specimen}/{label}/{sample}')
            raise e

    def __init__(self, specimen='train', root='..', hard_limit=None, dtype=np.float32):
        assert specimen in ('train', 'test', 'validation')
        self.root = root
        self.directory = os.path.join(root, 'dataset', 'OIDv6', specimen)
        classes = os.listdir(self.directory)
        self.descriptors: list[Descriptor] = []
        for class_name in classes:
            class_dir = os.path.join(self.directory, class_name, 'labels')
            for sample in os.listdir(class_dir):
                box = tuple(map(float, open(os.path.join(class_dir, sample)).read().rsplit(' ')[-4:]))
                descriptor: Descriptor = {'label': class_name, 'box': box, 'dir': (specimen, class_name, sample.rsplit('.', 1)[0])}
                self.descriptors.append(descriptor)
        self.shuffle()
        if hard_limit is not None: self.descriptors = self.descriptors[:hard_limit]
        self.dtype = dtype

    def shuffle(self):
        random.shuffle(self.descriptors)

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, idx):
        descriptor = self.descriptors[idx]
        return OIDv6Dataset.load_image(*descriptor['dir'], descriptor=descriptor, root=self.root, dtype=self.dtype)


class IMDataset(Dataset):
    available_categories = ('bus', 'car', 'motorcycle', 'truck')

    mappings = {

        # WRONG IMAGES
        # "n02892201": ("bus", "bus, autobus, coach, charabanc, double-decker, jitney, motorcoach, omnibus"),
        "n04487081": ("bus", "trolleybus, trolley coach, trackless trolle"),

        "n04467665": ("truck", "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi"),
        "n03417042": ("truck", "garbage truck, dustcart"),
        "n03930630": ("truck", "pickup, pickup truck"),
        # "n04467332": ("truck", "pickup, pickup truck"),
        # "n08641617": ("truck", "tow truck, tow car, wrecker"),
        "n02958343": ("car", "car, auto, automobile, machine, motorcar"),
        # "n03100232": ("car", "convertible"),
        "n03594945": ("car", "jeep, landrover"),
        "n03769881": ("car", "minivan"),
        "n04037443": ("car", "racer, race car, racing car"),
        "n04285008": ("car", "sports car, sport car"),
        "n03790512": ("motorcycle", "motorcycle, motorbike"),
        "n02835271": ("motorcycle", "bicycle-built-to-two, tandem bicycle, tandem"),
    }

    stratified_mappings = ["n04487081", "n04487081", "n04487081", "n04487081", "n04487081", "n04467665", "n04467665", "n04467665", "n04467665", "n04467665", "n02958343", "n03594945", "n03769881", "n04037443", "n04285008", "n03790512",
                           "n03790512", "n03790512", "n02835271", "n02835271", ]
    reverse = {
        "bus": ["n04487081"],
        "truck": ["n04467665"],
        "car": ["n02958343", "n03594945", "n03769881", "n04037443", "n04285008"],
        "motorcycle": ["n03790512", "n02835271"],
    }

    @staticmethod
    def load_image(
            specimen: str = 'train',
            mapping: str = None,
            sample: str = None,
            root: str = '..',
            log: bool = False,
            descriptor=None,
            dtype=np.float32,
    ):
        try:
            if mapping is None: mapping = random.choice(IMDataset.stratified_mappings)
            if mapping not in IMDataset.mappings: mapping = random.choice(IMDataset.reverse[mapping])
            assert specimen in ('train', 'test', 'validation')
            assert mapping in IMDataset.mappings
            directory = os.path.join(root, 'dataset', 'imagenet', mapping)  # TODO: Add TTV splits
            if sample is None: sample = random.choice(os.listdir(directory)).rsplit('.', 1)[0]
            if log: print(f'Loading {specimen}/{mapping}/{sample}')
            image = Image.open(os.path.join(directory, f'{sample}.JPEG'))
            if image.mode != 'RGB': image = image.convert('RGB')
            image = np.array(image, dtype=dtype)
            if descriptor is not None: return descriptor, image
            box = (0.0, 0.0, image.shape[1], image.shape[0])
            return {'label': IMDataset.mappings[mapping][0], 'box': box, 'dir': (specimen, mapping, sample)}, image
        except Exception as e:
            print(f'Error loading {specimen}/{mapping}/{sample}')
            raise e

    def __init__(self, specimen='train', root='..', hard_limit=None, dtype=np.float32):
        assert specimen in ('train', 'test', 'validation')
        self.root = root
        self.directory = os.path.join(root, 'dataset', 'imagenet')  # TODO add specimen
        mappings = os.listdir(self.directory)

        descriptors: dict[str, list[Descriptor]] = {}
        for mapping in mappings:
            class_dir = os.path.join(self.directory, mapping)
            for sample in os.listdir(class_dir):
                label = IMDataset.mappings[mapping][0]
                descriptor: Descriptor = {'label': label, 'box': None, 'dir': (specimen, mapping, sample.rsplit('.', 1)[0])}
                if label not in descriptors: descriptors[label] = []
                descriptors[label].append(descriptor)

        self.descriptors: list[Descriptor] = []
        if hard_limit is not None:
            while hard_limit > 0:
                for key in descriptors:
                    hard_limit -= 1
                    if len(descriptors[key]) > 0:
                        self.descriptors.append(descriptors[key].pop())
        else:
            for key in descriptors:
                self.descriptors.extend(descriptors[key])

        self.shuffle()
        self.dtype = dtype

    def shuffle(self):
        random.shuffle(self.descriptors)

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, idx):
        descriptor = self.descriptors[idx]
        return IMDataset.load_image(*descriptor['dir'], root=self.root, dtype=self.dtype)


def collate_fn(batch):
    if len(batch) == 1:
        return batch[0][0], batch[0][1]

    descriptors = [item[0] for item in batch]
    images = [item[1] for item in batch]

    images = default_collate(images)

    return descriptors, images


class PipelinedDataset_OLD(IterableDataset):
    def __init__(self, dir_dataset: Dataset,
                 process_sample: Callable[[Descriptor, np.ndarray], tuple[list[int], list[np.ndarray], Any]],
                 innate_batch=16, buffer_size=512):
        self.loader = DataLoader(dir_dataset, collate_fn=collate_fn)
        self.process_sample = process_sample
        self.innate_batch = innate_batch
        self.buf_desc = np.zeros((innate_batch, 5), dtype=np.float32)
        self.buf_img = np.zeros((innate_batch, 3, 32, 32), dtype=np.uint8)
        assert buffer_size % innate_batch == 0

    def flush_buffer(self):
        perm = np.random.permutation(len(self.buf_img))
        # descs = [self.buf_desc[i] for i in perm]
        descs = self.buf_desc[perm]
        imgs = self.buf_img[perm]

        for st in range(0, self.buf_img.shape[0], self.innate_batch):
            yield imgs[st:st + self.innate_batch], descs[st:st + self.innate_batch]

        self.buf_img.fill(0)

    def push_buffer(self, offset, labels, samples):
        if len(samples) + offset[0] > len(self.buf_img):

            fill_descs = labels[:len(self.buf_img) - offset[0]]
            fill_imgs = samples[:len(self.buf_img) - offset[0]]
            next_descs = labels[len(self.buf_img) - offset[0]:]
            next_imgs = samples[len(self.buf_img) - offset[0]:]

            self.buf_desc[offset[0]:] = fill_descs
            self.buf_img[offset[0]:] = fill_imgs
            for x in self.flush_buffer(): yield x

            offset[0] = 0
            yield from self.push_buffer(offset, next_descs, next_imgs)
            return

        self.buf_desc[offset[0]:offset[0] + len(labels)] = labels
        self.buf_img[offset[0]:offset[0] + len(samples)] = samples

        if offset[0] == len(self.buf_img):
            for x in self.flush_buffer(): yield x
            offset[0] = 0
            return

        offset[0] = offset + len(samples)

    def __iter__(self):
        offset = [0]
        for descriptor, image in self.loader:
            labels, samples, *_ = self.process_sample(descriptor, image)
            yield from self.push_buffer(offset, labels, samples)


class PipelinedDataset(IterableDataset):
    def __init__(self,
                 dir_dataset: Dataset,
                 process_sample: Callable[[Descriptor, np.ndarray], tuple[list[int], list[np.ndarray], Any]],
                 skip_factor=0.85
                 ):
        self.loader = DataLoader(dir_dataset, collate_fn=collate_fn)
        self.process_sample = process_sample
        self.skip_factor = skip_factor

    def __iter__(self):
        for descriptor, image in self.loader:
            labels, samples, *_ = self.process_sample(descriptor, image)
            for label, sample in zip(labels, samples):
                if label.argmax() == 4 and random.random() < self.skip_factor: continue
                yield sample / 255, label


class AlternatingDataset(IterableDataset):
    def __init__(self, d1: Dataset, d2: Dataset):
        self.d1 = d1
        self.d2 = d2
        self.hic = True

    def __iter__(self):
        for d1, d2 in zip(self.d1, self.d2):
            yield d1
            yield d2
