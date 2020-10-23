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
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        return x

    def get_prediction(self, img, labels, support_imgs=None, metric_func='l2'):
        if support_set is not None:
            self.embeddings = self.embedding.predict(support_imgs)

        emb = self.embedding.predict(img)
        distances = [
            np.mean(np.square(emb - e)) \
            for e in self.embeddings
        ]

        pred = distances.index(min(distances))
        return labels[pred]


    def l2_loss(self, inputs):
        a, b = inputs
        return K.sum(
            K.square(a - b[:, 0]),
            axis=1,
            keepdims=True,
        )


    def build_main_model(self):
        images = Input(self.input_shape)
        labels = Input((1,))
        embedding = self.feature_extractor(images)

        train_model = Model(inputs=[images, labels],
                            outputs=[embedding])
        train_model.compile(optimizer=Adam(self.lr),
                            loss=tfa.losses.TripletSemiHardLoss(),
                            metrics=['accuracy'])
        return train_model


    def embedding_model(self):
            return Model(
                inputs=self.main_model.inputs[0],
                outputs= self.main_model.outputs,
                name="embbeding",
            )


    def train_one_epoch(self, data_gen):
        total_loss = []
        for x, y in batch_gen.next_batch():
            total_loss.append(self.main_model.train_on_batch(x, y))

        return np.mean(np.array(total_loss), axis=0)


    @staticmethod
    def init_hist():
        return {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": [],
        }


    def train(self, data_gen, epochs=50):
        print("Train autoencoder model")
        print("Train on {} samples".format(len(data_gen.x)))
        history = self.init_hist()

        for e in range(epochs):
            start_time = datetime.datetime.now()
            print("Train epochs {}/{} - ".format(e + 1, epochs), end="")

            batch_loss = self.init_hist()

            for x, y in data_gen.next_batch():
                loss, _,_,acc,_ = self.main_model.train_on_batch(x, y)
                batch_loss['loss'].append(loss)

            # evaluation
            # batch_loss['val_loss'] = self.main_model.evaluate(test_gen.x_test,
            #                                                   test_gen.y_test,
            #                                                   verbose=False)

            mean_loss = np.mean(np.array(batch_loss['loss']))
            # mean_val_loss = np.mean(np.array(batch_loss['val_loss']))

            history['loss'].append(mean_loss)
            # history['val_loss'].append(mean_val_loss)

            print("Loss: {}, Val Loss: {} - {}".format(
                mean_loss, 0,
                datetime.datetime.now() - start_time
            ))

        self.history = history
        return history


    