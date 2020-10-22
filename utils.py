import pickle
import numpy as np
import urllib.request
import keras.preprocessing.image as image_processing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow

MEAN_PIXCELS = np.array([103.939, 116.779, 123.68])
DECOMPOSERS = {
    'pca': PCA(),
    'tsne': TSNE()
}

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


def visualize_scatter_with_images(X_2d_data, images, figsize=(10,10), image_zoom=0.5):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()


def visualize_scatter(data_2d, label_ids, figsize=(8,8), legend=True,title="None"):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))

    for i,label_id in enumerate(np.unique(label_ids)):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=colors[i],
                    linewidth='1',
                    alpha=0.8,
                    label=label_id)
    if legend:
        plt.legend(loc='best')
    else:
        # plt.title(title)
        plt.axis('off')

    plt.show()


def scatter_plot(x, y, encoder, name='chart', opt='pca', plot_img=None,
                legend=True, title="None"):
    step = 1
    if encoder.input_shape[-1] != x.shape[-1]:
        x = triple_channels(x)

    x_embeddings = encoder.predict(x)
    if len(x_embeddings.shape) > 2:
        x_embeddings = x_embeddings.reshape(x_embeddings.shape[0], -1)
    decomposed_embeddings = DECOMPOSERS[opt].fit_transform(x_embeddings)
    if plot_img:
        return visualize_scatter_with_images(decomposed_embeddings,x)
    visualize_scatter(decomposed_embeddings, y, legend=legend,title=title)