import tensorflow as tf
import tensorflow.keras.backend as K
import datetime
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    Lambda,
    Activation,
    BatchNormalization,
    Conv2DTranspose,
    Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_addons as tfa
from scipy.spatial.distance import cosine
import numpy as np


class FaceModel:
    def __init__(self, rst, num_of_classes, lr=1e-3, feat_dims=128):
        self.rst = rst
        self.lr = lr
        self.num_of_classes = num_of_classes
        self.input_shape = (self.rst, self.rst, 3)
        self.feat_dims = feat_dims

        self.main_model = self.build_main_model()
        self.embedding = self.embedding_model()

    def feature_extractor(self, image):
        vgg16 = VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=Input(self.input_shape),
            input_shape=self.input_shape,
        )

        x = vgg16(image)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.feat_dims)(x)
        # normalize
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        return x

    def get_prediction(self, img, labels, support_imgs=None, metric_func="l2"):
        if support_imgs is not None:
            self.embeddings = self.embedding.predict(support_imgs)

        emb = self.embedding.predict(img)
        distances = [1 - cosine(emb[0], e) for e in self.embeddings]

        pred_prob = max(distances)
        pred = distances.index(pred_prob)
        return labels[pred], pred_prob

    def build_main_model(self):
        images = Input(self.input_shape)
        embedding = self.feature_extractor(images)

        train_model = Model(inputs=[images], outputs=[embedding])
        train_model.compile(
            optimizer=Adam(self.lr), loss=tfa.losses.TripletSemiHardLoss(),
        )
        return train_model

    def embedding_model(self):
        return Model(
            inputs=self.main_model.inputs[0],
            outputs=self.main_model.outputs,
            name="embbeding",
        )

    def train(self, data_gen, epochs=5):
        # simply fit the model
        tfgen = (
            tf.data.Dataset.from_tensor_slices((data_gen.x, data_gen.y))
            .repeat()
            .shuffle(1024)
            .batch(64)
        )

        self.history = self.main_model.fit(tfgen, steps_per_epoch=32, epochs=epochs)

    def calculate_embeddings(self, x, y):
        self.embeddings = self.embedding.predict(x)
        self.support_labels = y

    def evaluate(self, x_test, y_test):
        """
        Evaluate model perfomance by accracy metric

        @returns: accuracy, pred_wrong_indices
        """
        if not hasattr(self, "embeddings"):
            raise ("embeddings is not calculated")

        preds = []
        for x in x_test:
            pred, _ = self.get_prediction(np.expand_dims(x, 0), self.support_labels)
            preds.append(pred)

        preds = np.array(preds)
        pred_boolean = preds == y_test
        pred_wrong = np.where(pred_boolean == False)[0]
        return pred_boolean.mean(), pred_wrong


class AutoEncoder:
    def __init__(self, lr, rst, latent_dim=128):
        self.rst = rst
        self.input_shape = (rst, rst, 3)
        self.latent_dim = latent_dim
        self.build_network()
        self.model.compile(optimizer=Adam(lr=lr), loss="mse")

    def _build_encoder(self):
        inp = Input(shape=self.input_shape)
        x = inp
        filters = [64, 128, 256]
        for f in filters:
            x = Conv2D(f, kernel_size=3, strides=2, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)

        # init shape for decoder
        self.init_shape = K.int_shape(x)
        latent = Flatten()(x)
        latent = Dense(self.latent_dim)(latent)
        self.encoder = Model(inputs=inp, outputs=latent, name="encoder")

    def build_network(self):
        self._build_encoder()
        _, h, w, c = self.init_shape
        filters = [256, 128, 64]

        # decoder
        inp = Input(shape=self.input_shape)
        x = self.encoder(inp)
        x = Dense(h * w * c)(x)
        x = Reshape(self.init_shape[1:])(x)
        for f in filters:
            x = Conv2DTranspose(f, kernel_size=3, strides=2, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)

        out = Conv2D(3, kernel_size=3, strides=1, padding="same", activation="tanh")(x)

        self.model = Model(inputs=inp, outputs=out, name="base_model")

    def train(self, data_gen, epochs=5):
        tfgen = (
            tf.data.Dataset.from_tensor_slices((data_gen.x, data_gen.x))
            .repeat()
            .shuffle(1024)
            .batch(64)
        )

        self.history = self.model.fit(tfgen, steps_per_epoch=32, epochs=epochs)
