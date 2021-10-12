import pickle

PATH = '/data/aad/image_datasets/nb201_new/NAS-BENCH-102-4-v1.0-archive/'


def get_id(arch):
    with open('arch_to_id.pkl', 'rb') as f:
        data = pickle.load(f)
    return data[arch]

def check_seed(arch_id):
    with open('imagenet.pkl', 'rb') as f:
        d = pickle.load(f)
    return arch_id in d

if __name__ == '__main__':

    #pcdarts
    #arch = '|nor_conv_3x3~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|skip_connect~0|avg_pool_3x3~1|avg_pool_3x3~2|'
    arch = '|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_1x1~2|'
    print(get_id(arch)[0])

    print(check_seed(get_id(arch)[0]))


