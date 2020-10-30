import pickle
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras.preprocessing.image as image_processing
import cv2
import face_localization
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow

MEAN_PIXCELS = np.array([103.939, 116.779, 123.68])
DECOMPOSERS = {"pca": PCA(), "tsne": TSNE()}


def pickle_save(object, path):
    try:
        print("save data to {} successfully".format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print("save data to {} failed".format(path))


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


def _processing(img, normalize, preprcs):
    if preprcs:
        img = preprocess(img)

    if normalize:
        img = norm(img)

    return img


def get_image_http(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def readimg(path, get_face=True, normalize=True, preprcs=True, size=64):
    """
    @returns: image, bbox, face
    """
    bbox = None
    face = None
    default = None, None, None
    try:
        if path.startswith("http") or path.startswith("base"):
            img = get_image_http(path)
        else:
            img = cv2.imread(path)
    except Exception as e:
        print("Could not read img, ERROR: {}".format(str(e)))
        return default

    if img is None:
        return default

    if get_face:
        face = face_localization.extract_face(img, True)
        if face is None:
            return default

        face, bbox = face
        face = _processing(face, normalize, preprcs)
        face = cv2.resize(face, (size, size))
    else:
        img = _processing(img, normalize, preprcs)
        img = cv2.resize(img, (size, size))

    return img, bbox, face


def draw_bbox(img, coordinates, text="face", color=(0, 0, 0)):
    "The pixcel's range should be [0, 255]"
    x, y, w, h = coordinates
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(img, (x, y), (x + w, y - 25), color, -1)
    return cv2.putText(img, text, (x + w, y - 10), 0, 0.5, (255, 255, 255))


def transform(x, seed=0):
    np.random.seed(seed)
    img = image_processing.random_rotation(x, 0.2)
    img = image_processing.random_shear(img, 30)
    img = image_processing.random_zoom(img, (0.5, 1.1))
    if np.random.rand() >= 0.5:
        img = np.fliplr(img)

    return img


def show_images(img_array, denorm=True, deprcs=False):
    try:
        shape = img_array.shape
        img_array = img_array.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))
        # convert 1 channel to 3 channels
        channels = img_array.shape[-1]
        resolution = img_array.shape[2]
        img_rows = img_array.shape[0]
        img_cols = img_array.shape[1]

        img = np.full([resolution * img_rows, resolution * img_cols, channels], 0.0)
        for r in range(img_rows):
            for c in range(img_cols):
                img[
                    (resolution * r) : (resolution * (r + 1)),
                    (resolution * (c % 10)) : (resolution * ((c % 10) + 1)),
                    :,
                ] = img_array[r, c]

        if denorm:
            img = de_norm(img)
        if deprcs:
            img = deprocess(img)

        cv2_imshow(img)
    except Exception as e:
        print("Could not show image data, ERROR: {}".format(str(e)))


def make_border(img, color, bordersize=3):
    return cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )


def visualize_scatter_with_images(
    X_2d_data, images, labels, figsize=(10, 10), image_zoom=0.5
):
    if type(labels[0]) is not int:
        labels = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    # convert RBG -> BGR, change range from [-1, 1] to [0, 1]

    images = de_norm(images[..., [2, 1, 0]]) / 255.0

    for xy, i, cl in zip(X_2d_data, images, labels):
        x0, y0 = xy
        i = make_border(i, colors[cl], 4)
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()


def visualize_scatter(data_2d, label_ids, figsize=(8, 8), legend=True, title="None"):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))
    colors = cm.rainbow(np.linspace(0, 1, nb_classes))

    for i, label_id in enumerate(np.unique(label_ids)):
        plt.scatter(
            data_2d[np.where(label_ids == label_id), 0],
            data_2d[np.where(label_ids == label_id), 1],
            marker="o",
            color=colors[i],
            linewidth="1",
            alpha=0.8,
            label=label_id,
        )
    if legend:
        plt.legend(loc="best")
    else:
        # plt.title(title)
        plt.axis("off")

    plt.show()


def scatter_plot(
    x,
    y,
    encoder,
    name="chart",
    opt="pca",
    plot_img=None,
    legend=True,
    title="None",
    figsize=(10, 10),
    image_zoom=0.5,
):
    x_embeddings = encoder.predict(x)
    if len(x_embeddings.shape) > 2:
        x_embeddings = x_embeddings.reshape(x_embeddings.shape[0], -1)
    decomposed_embeddings = DECOMPOSERS[opt].fit_transform(x_embeddings)

    if plot_img:
        assert opt == "tsne"
        return visualize_scatter_with_images(
            decomposed_embeddings, x, y, figsize, image_zoom
        )

    visualize_scatter(decomposed_embeddings, y, legend=legend, title=title)


def split_by_label(x, y, test_size=0.3):
    classes = np.unique(y)
    np.random.shuffle(classes)
    for_test = int(len(classes) * test_size)
    ids = np.arange(len(x))
    per_class_ids = {c: ids[y == c] for c in classes}

    to_train_idx = np.concatenate([per_class_ids[c] for c in classes[for_test:]])
    to_test_idx = np.concatenate([per_class_ids[c] for c in classes[:for_test]])

    return (x[to_train_idx], x[to_test_idx], y[to_train_idx], y[to_test_idx])


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
