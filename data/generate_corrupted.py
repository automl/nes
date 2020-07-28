import sys
import torchvision.datasets as dset
import torchvision.transforms as trn
import numpy as np

from pathlib import Path
from tqdm import tqdm

from data.corruptions import *


DATA_PATH = 'data'

d = {}
d['Gaussian Noise'] = gaussian_noise
d['Shot Noise'] = shot_noise
d['Impulse Noise'] = impulse_noise
d['Defocus Blur'] = defocus_blur
d['Glass Blur'] = glass_blur
d['Motion Blur'] = motion_blur
d['Zoom Blur'] = zoom_blur
d['Snow'] = snow
d['Frost'] = frost
d['Fog'] = fog
d['Brightness'] = brightness
d['Contrast'] = contrast
d['Elastic'] = elastic_transform
d['Pixelate'] = pixelate
d['JPEG'] = jpeg_compression

d['Speckle Noise'] = speckle_noise
d['Gaussian Blur'] = gaussian_blur
d['Spatter'] = spatter
d['Saturate'] = saturate

if __name__ == '__main__':

    # run the corrupted dataset generation in parallel
    # for every corruption type
    corruptions_list = [x for x in d.keys()]
    method_idx = sys.argv[1]
    method_name = corruptions_list[int(method_idx)]

    # The bit below applies the distortion functions defined above to 
    # CIFAR-10 and saves the new distorted datapoints at the locations specified. 

    for train_split in [True, False]:
        dataset = dset.CIFAR10(DATA_PATH, train=train_split, download=False)
        convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

        print('Creating images for the corruption', method_name)

        for severity in tqdm(range(1, 6), desc='severity'):
            cifar_c = []
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img in tqdm(dataset.data):
                cifar_c.append(np.uint8(corruption(convert_img(img))))

            labels = dataset.targets

            images_path = DATA_PATH + '/cifar10-C/' + d[
                method_name].__name__ + f'/severity_{severity}'
            labels_path = DATA_PATH + '/cifar10-C/' + d[
                method_name].__name__ + f'/severity_{severity}'

            Path(images_path).mkdir(parents=True, exist_ok=True)
            Path(labels_path).mkdir(parents=True, exist_ok=True)

            np.save(images_path + f'/images_{"train" if train_split else "test"}.npy',
                    np.array(cifar_c).astype(np.uint8))

            np.save(labels_path + f'/labels_{"train" if train_split else "test"}.npy',
                    np.array(labels))

