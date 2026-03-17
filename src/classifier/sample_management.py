from .dataset_management import Box, Descriptor, area, class_to_idx
import numpy as np
import cv2


def pad_up_sample(descriptor: Descriptor, image: np.ndarray, sampling=500):
    """
    Add Padding around the image to make its width and height match any multiple of sampling.
    """
    sampling = sampling // 2
    h, w = image.shape[:2]
    hn, wn = (1 + (h - 1) // sampling) * sampling, (1 + (w - 1) // sampling) * sampling
    padded = np.zeros((hn, wn, image.shape[2]), dtype=image.dtype)
    ho, wo = (hn - h) // 2, (wn - w) // 2
    padded[ho:ho + h, wo:wo + w, :] = image
    descriptor, box = descriptor.copy(), descriptor['box']
    descriptor['box'] = (wo + box[0], ho + box[1], wo + box[2], ho + box[3])
    return descriptor, padded


def rescale_sample(descriptor: Descriptor, image: np.ndarray, target: int | float | tuple[int, int]):
    """
    Smart, context-based, and aspect-preserving rescaling of the image
    """
    if isinstance(target, float):
        target = (int(image.shape[0] * target), int(image.shape[1] * target))
    if isinstance(target, int):
        target = (target, int(target / image.shape[0] * image.shape[1]))
    rescaled = cv2.resize(image, dsize=target[::-1], interpolation=cv2.INTER_CUBIC)
    rescaled = np.clip(rescaled, 0, 255)
    descriptor, box = descriptor.copy(), descriptor['box']
    descriptor['box'] = (box[0] / image.shape[1] * target[1], box[1] / image.shape[0] * target[0],
                         box[2] / image.shape[1] * target[1], box[3] / image.shape[0] * target[0])
    return descriptor, rescaled


def ensure_within(descriptor: Descriptor, image: np.ndarray, max_bounds=256):
    """
    Since the max image size is given in the assignment, let's ensure our images fit within the max_bounds-by-max_bounds square
    """
    if image.shape[0] > image.shape[1]:
        return rescale_sample(descriptor, image, max_bounds)
    return rescale_sample(descriptor, image, int(max_bounds / image.shape[1] * image.shape[0]))


def context_crop_sample(descriptor: Descriptor, image: np.ndarray, ratio=1.0):
    """
    Crop away areas too far from the main item of the image, as described by the descriptor.
    """
    h, w = image.shape[:2]
    descriptor, box = descriptor.copy(), descriptor['box']
    major_size = (box[2] - box[0] + box[3] - box[1]) * ratio / 2
    bx1 = int(max(0., box[0] - major_size))
    bx2 = int(min(w, box[2] + major_size))
    by1 = int(max(0., box[1] - major_size))
    by2 = int(min(h, box[3] + major_size))
    cropped = image[by1:by2, bx1:bx2, :]
    descriptor['box'] = (box[0] - bx1, box[1] - by1, box[2] - bx1, box[3] - by1)
    return descriptor, cropped


def center_crop_sample(descriptor: Descriptor, image: np.ndarray, ratio=1.0):
    h, w = image.shape[:2]
    descriptor, box = descriptor.copy(), descriptor['box']
    center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    major_size = (box[2] - box[0] + box[3] - box[1]) * ratio / 2
    bx1 = int(max(0., center[0] - major_size))
    bx2 = int(min(w, center[0] + major_size))
    by1 = int(max(0., center[1] - major_size))
    by2 = int(min(h, center[1] + major_size))
    cropped = image[by1:by2, bx1:bx2, :]
    descriptor['box'] = (box[0] - bx1, box[1] - by1, box[2] - bx1, box[3] - by1)
    return descriptor, cropped



def clip(x, m, M):
    return max(m, min(M, x))


def split_sample(descriptor: Descriptor, image: np.ndarray, sampling=500, return_boxes=False):
    """
    Generate windows of size sampling x sampling from the image, with a striding of sampling // 2.
    """
    count = 0
    slide = sampling // 2

    h, w = image.shape[:2]
    while w >= sampling and h >= sampling:
        count += (w // slide - 1) * (h // slide - 1)
        w, h = w // 2, h // 2

    samples = np.zeros((count, 3, sampling, sampling), dtype=image.dtype)
    labels = np.zeros((count, 5), dtype=image.dtype)

    x1, y1, x2, y2 = descriptor['box']
    original_area = area(descriptor['box'])
    is_known_label = descriptor['label'] in ["bus", "truck", "car", "bike"]
    l_desc = []

    h, w = image.shape[:2]
    c, s = 0, 1
    while w - s * sampling + 1 > 0 and h - s * sampling + 1 > 0:
        for x in range(0, w - s * sampling + 1, s * slide):
            for y in range(0, h - s * sampling + 1, s * slide):
                samples[c] = image[y:y + s * sampling:s, x:x + s * sampling:s].transpose(2, 0, 1)

                raw_box = ((x1 - x) / s, (y1 - y) / s,
                           (x2 - x) / s, (y2 - y) / s)
                box = (clip(raw_box[0], 0, sampling), clip(raw_box[1], 0, sampling),
                       clip(raw_box[2], 0, sampling), clip(raw_box[3], 0, sampling))
                # min(raw_box[0] + sampling, raw_box[2]), min(raw_box[1] + sampling, raw_box[3]))

                if is_known_label:
                    ratio = clip(max(
                        area(box) / original_area * s * s,
                        area(box) / sampling / sampling),
                        0, 1
                    )
                    # ratio = min(0, max(
                    #     area(box) / original_area, 1
                    #     # area(box) / area(raw_box),
                    # ))
                    labels[c, class_to_idx[descriptor['label']]] = ratio
                    labels[c, class_to_idx['none']] = 1 - ratio
                else:
                    labels[c, class_to_idx['none']] = 1

                if return_boxes:
                    desc = descriptor.copy()
                    desc['box'] = box
                    l_desc.append(desc)
                c += 1
                if c == 102:
                    continue
        s = s * 2

    return labels, samples, l_desc


def process_sample(descriptor, image, return_boxes=False):
    """
    Aggregate all the operations above in a single function.
    """
    descriptor, image = context_crop_sample(descriptor, image, ratio=0.6)
    descriptor, image = ensure_within(descriptor, image, max_bounds=256)
    descriptor, image = pad_up_sample(descriptor, image, sampling=32)
    labels, samples, descriptors = split_sample(descriptor, image, sampling=32, return_boxes=return_boxes)
    return labels, samples, descriptors, descriptor, image


def process_sample_v2(descriptor, image, *args):
    descriptor, image = center_crop_sample(descriptor, image, ratio=0.6)  # Very focussed
    descriptor, image = ensure_within(descriptor, image, max_bounds=244)  # For mobile net
    descriptor, image = pad_up_sample(descriptor, image, sampling=244 * 2)  # For mobile net
    labels = np.zeros((1, 2), dtype=np.int64)
    labels[0, 0] = 1 if descriptor['label'] in ["bus", "truck", "car", "motorcycle"] else 0
    labels[0, 1] = class_to_idx[descriptor['label']] if descriptor['label'] in ["bus", "truck", "car", "motorcycle"] else 0
    return labels, [image.transpose(2, 0, 1)], None, descriptor, image

def simple_process_sample(descriptor, image, image_size=244):
    descriptor, image = ensure_within(descriptor, image, max_bounds=image_size)
    descriptor, image = pad_up_sample(descriptor, image, sampling=2 * image_size)
    labels = np.zeros((1, 2), dtype=np.int64)
    labels[0, 0] = 1 # is known or not
    labels[0, 1] = class_to_idx[descriptor['label']]
    return labels, [image.transpose(2, 0, 1)], None, None, None