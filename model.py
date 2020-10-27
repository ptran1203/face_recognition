import tensorflow as tf
import tensorflow.keras.backend as K
import datetime
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Input, Conv2D, GlobalAveragePooling2D,
    Dense, Embedding, Lambda, Activation
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
            weights='imagenet',
            input_tensor=Input(self.input_shape),
            input_shape=self.input_shape,
        )

        x = vgg16(image)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.feat_dims)(x)
        # normalize
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        return x


    def get_prediction(self, img, labels, support_imgs=None, metric_func='l2'):
        if support_imgs is not None:
            self.embeddings = self.embedding.predict(support_imgs)

        emb = self.embedding.predict(img)
        distances = [
            1 - cosine(emb[0], e)
            for e in self.embeddings
        ]

        pred_prob = max(distances)
        pred = distances.index(pred_prob)
        return labels[pred], pred_prob


    def build_main_model(self):
        images = Input(self.input_shape)
        embedding = self.feature_extractor(images)

        train_model = Model(inputs=[images],
                            outputs=[embedding])
        train_model.compile(optimizer=Adam(self.lr),
                            loss=tfa.losses.TripletSemiHardLoss(),
                            )
        return train_model



    def embedding_model(self):
            return Model(
                inputs=self.main_model.inputs[0],
                outputs= self.main_model.outputs,
                name="embbeding",
            )


    def train(self, data_gen, epochs=5):
        # simply fit the model
        tfgen = tf.data.Dataset.from_tensor_slices((data_gen.x, data_gen.y)). \
                            repeat(). \
                            shuffle(1024).batch(64)

        self.history = self.main_model.fit(tfgen, steps_per_epoch=32, epochs=epochs)

    
    def calculate_embeddings(self, x, y):
        self.embeddings = self.embedding.predict(x)
        self.support_labels = y


    def evaluate(self, x_test, y_test):
        if not hasattr(self, 'embeddings'):
            raise("embeddings is not calculated")

        preds = []
        for x in x_test:
            pred, _ = self.get_prediction(np.expand_dims(x,0), self.support_labels)
            preds.append(pred)

        preds = np.array([preds])
        return (preds == y_test).mean()





    