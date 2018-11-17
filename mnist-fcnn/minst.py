import gzip
import shutil
import os
import urllib.request


def download(url, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    urllib.request.urlretrieve(url, path + '.gz')
    with gzip.open(path + '.gz', 'rb') as f_in:
        with open(path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(path + '.gz')


urls = ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

names = ('train-images.idx3-ubyte',
         'train-labels.idx1-ubyte',
         't10k-images.idx3-ubyte',
         't10k-labels.idx1-ubyte')

for url, name in zip(urls, names):
    download(url, name, "mnist")
