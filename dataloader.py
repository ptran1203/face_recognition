import numpy as np
import seaborn as sns
import utils
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

sns.set_theme()


class DataGenerator:
    def __init__(
        self,
        data_path,
        batch_size=64,
        img_resolution=128,
        split_option=1,
        test_size=0.3,
        kshot=5,
        force_reload_data=False,
    ):

        SPLIT_BY_LABEL = 1
        NORMAL_SPLIT = 2

        self.img_resolution = img_resolution
        self.batch_size = batch_size
        self.force_reload_data = force_reload_data
        self.loaddata(data_path)

        self.init_labels = np.copy(self.labels)

        # filter identities have more than 1 image
        self.x, self.labels = self.filter_one_image(self.x, self.labels)

        self.x = utils.norm(self.x)

        self.x, self.x_test, self.labels, self.labels_test = (
            utils.split_by_label(self.x, self.labels, test_size=test_size)
            if split_option == SPLIT_BY_LABEL
            else train_test_split(self.x, self.labels, test_size=test_size)
        )

        (
            self.x_test,
            self.x_support,
            self.labels_test,
            self.labels_support,
        ) = self.support_split(self.x_test, self.labels_test, kshot=kshot)

        # convert string label to numberical label
        _, self.y = np.unique(self.labels, return_inverse=True)
        _, self.y_test = np.unique(self.labels_test, return_inverse=True)
        _, self.y_support = np.unique(self.labels_support, return_inverse=True)

        self.classes = np.unique(self.y)
        self.per_class_ids = {}
        ids = np.arange(len(self.x))
        for c in self.classes[:3]:
            self.per_class_ids[c] = ids[self.y == c]
            utils.show_images(self.x[self.per_class_ids[c]][:10], True, False)

    @staticmethod
    def filter_one_image(x, y):
        # filter identities have more than 1 image
        counter = Counter(y)
        filtered = [c for c in counter if counter[c] > 1]
        indices = np.where(np.in1d(y, filtered))

        return x[indices], y[indices]

    @staticmethod
    def support_split(x, y, kshot=5):
        """
        Randomly picks <kshot> images from each class
        """
        classes = np.unique(y)
        ids = np.arange(len(x))
        per_class_ids = [ids[y == c] for c in classes]
        selected = np.concatenate([x[:kshot] for x in per_class_ids])
        remains = np.setdiff1d(ids, selected)

        return x[remains], x[selected], y[remains], y[selected]

    def get_data_for_class(self, classid):
        return self.x[self.per_class_ids[classid]]

    def loaddata(self, data_path):
        temp_file_name = "./temp_data.pkl"
        if not self.force_reload_data and os.path.isfile(temp_file_name):
            self.x, self.labels = utils.pickle_load(temp_file_name)
        elif data_path.endswith(".pkl"):
            self.x, self.labels = utils.pickle_load(data_path)
        else:
            print("Read data from directory")
            count = 0
            labels = []
            imgs = []
            for sub_dir in os.listdir(data_path):
                dir_ = os.path.join(data_path, sub_dir)
                if not os.path.isdir(dir_):
                    continue
                fnames = os.listdir(dir_)
                print("-> {} total: {}: ".format(sub_dir, len(fnames), end=""))
                icount = 0
                for fname in fnames:
                    icount += 1
                    # Get the face image
                    _, _, img = utils.readimg(
                        os.path.join(dir_, fname),
                        get_face=True,
                        normalize=False,
                        preprcs=False,
                        size=self.img_resolution,
                    )
                    if img is not None:
                        imgs.append(img)
                        labels.append(sub_dir)
                    print("{}, ".format(icount), end="")
                print("")
                count += icount

            print("Done, {}/{} images were loaded".format(len(imgs), count))
            self.x = np.array(imgs)
            self.labels = np.array(labels)

            utils.pickle_save((self.x, self.labels), temp_file_name)

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

        for start_idx in range(
            0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size
        ):
            access_pattern = indices[start_idx : start_idx + self.batch_size]
            batch_y = [onehot_labels[access_pattern], self.dummy]
            yield (
                [dataset_x[access_pattern, :, :, :], labels[access_pattern]],
                batch_y,
            )
