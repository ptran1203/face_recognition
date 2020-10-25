import numpy as np
import utils
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class DataGenerator:
    def __init__(self, data_path, batch_size=64, img_resolution=128):
        self.img_resolution = img_resolution
        self.batch_size = batch_size
        self.loaddata(data_path)
        utils.show_images(self.x[:10], False, False)

        # filter identities have more than 1 image
        self.x, self.labels = self.filter_one_image(self.x, self.labels)
        self.x = utils.norm(self.x)

        self.x, self.x_test, self.labels, self.labels_test = train_test_split(self.x, self.labels, test_size=0.3)

        self.x_test, self.labels_test = self.filter_one_image(self.x_test, self.labels_test)

        _, self.y = np.unique(self.labels, return_inverse=True)
        _, self.y_test = np.unique(self.labels_test, return_inverse=True)

        print(self.labels.shape, self.y.shape)

        self.classes = np.unique(self.y)
        self.per_class_ids = {}
        ids = np.array(range(len(self.x)))
        for c in self.classes:
            self.per_class_ids[c] = ids[self.y == c]


    @staticmethod
    def filter_one_image(x, y):
        # filter identities have more than 1 image
        counter = Counter(y)
        filtered = ([c for c in counter if counter[c] > 1])
        indices = np.where(np.in1d(y, filtered))

        return x[indices], y[indices]

    
    def loaddata(self, data_path):
        if data_path.endswith(".pkl"):
            self.x, self.labels = utils.pickle_load(data_path)
        else:
            print("Read data from directory")
            labels = []
            imgs = []
            for sub_dir in os.listdir(data_path):
                dir_ = os.path.join(data_path, sub_dir)
                for fname in os.listdir(dir_):
                    # Get the face image
                    _, _, img = utils.readimg(os.path.join(dir_, fname),
                                          extract_face=True,
                                          normalize=False,
                                          preprcs=False,
                                          size=self.img_resolution)
                    if img is not None:
                        imgs.append(img)
                        labels.append(sub_dir)

            print("Done, {} images were loaded".format(len(imgs)))
            self.x = np.array(imgs)
            self.labels = np.array(labels)


    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size
        try:
            np.random.shuffle(self.per_class_ids[c])
            to_return = self.per_class_ids[c][0:samples]
            return self.dataset_x[to_return]
        except:
            random = np.arange(self.dataset_x.shape[0])
            np.random.shuffle(random)
            to_return = random[:samples]
            return self.dataset_x[to_return]


    def augment_one(self, x, y):
        seed = np.random.randint(0, 100)
        new_x = utils.transform(x, seed)
        new_y = utils.transform(y, seed)
        return new_x, new_y


    def augment_array(self, x, y, augment_factor):
        imgs = []
        labels = []
        for i in range(len(x)):
            imgs.append(x[i])
            labels.append(y[i])
            for _ in range(augment_factor):
                _x, _y = self.augment_one(x[i], y[i])
                imgs.append(_x)
                labels.append(_y)

        return np.array(imgs), np.array(labels)


    def next_batch(self):
        dataset_x = self.x
        labels = self.y
        onehot_labels = to_categorical(labels, len(self.classes))

        indices = np.arange(dataset_x.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            batch_y = [onehot_labels[access_pattern], self.dummy]
            yield (
                [dataset_x[access_pattern, :, :, :], labels[access_pattern]],
                batch_y,
            )
