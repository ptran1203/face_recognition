import pickle
import numpy as np
import urllib.request
import keras.preprocessing.image as image_processing
import cv2
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow

MEAN_PIXCELS = np.array([103.939, 116.779, 123.68])
def pickle_save(object, path):
    try:
        print('save data to {} successfully'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print('save data to {} failed'.format(path))


def pickle_load(path):
    try:
        # print("Loading data from {} - ".format(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        print(str(e))
        return None

def norm(imgs):
    return (imgs - 127.5) / 127.5


def de_norm(imgs):
    return imgs * 127.5 + 127.5


def preprocess(imgs):
    """
    BGR -> RBG then subtract the mean
    """
    return imgs - MEAN_PIXCELS


def deprocess(imgs):
    return imgs + MEAN_PIXCELS


def transform(x, seed=0):
    np.random.seed(seed)
    img = image_processing.random_rotation(x, 0.2)
    img = image_processing.random_shear(img, 30)
    img = image_processing.random_zoom(img, (0.5, 1.1))
    if np.random.rand() >= 0.5:
        img = np.fliplr(img)

    return img


def show_images(img_array, denorm=True, deprcs=True):
    shape = img_array.shape
    img_array = img_array.reshape(
        (-1, shape[-4], shape[-3], shape[-2], shape[-1])
    )
    # convert 1 channel to 3 channels
    channels = img_array.shape[-1]
    resolution = img_array.shape[2]
    img_rows = img_array.shape[0]
    img_cols = img_array.shape[1]

    img = np.full([resolution * img_rows, resolution * img_cols, channels], 0.0)
    for r in range(img_rows):
        for c in range(img_cols):
            img[
            (resolution * r): (resolution * (r + 1)),
            (resolution * (c % 10)): (resolution * ((c % 10) + 1)),
            :] = img_array[r, c]

    if denorm:
        img = de_norm(img)
    if deprcs:
        img = deprocess(img)

    cv2_imshow(img)

def http_get_img(url, rst=64, gray=False, normalize=True):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    if rst is not None:
        img = cv2.resize(img, (rst, rst))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = np.expand_dims(img, 0)
    if normalize:
        img = norm(preprocess(img))

    return img