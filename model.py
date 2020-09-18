import keras
import datetime
from keras.applications.vgg16 import VGG16
from keras.layers import (
    Input, Conv2D, GlobalAveragePooling2D,
    Dense, Embedding
)


class FaceModel:
    def __init__(self, rst, num_of_classes, lr=1e-3, feat_dims=128):
        self.rst = rst
        self.lr = lr
        self.num_of_classes = num_of_classes
        self.input_shape = (self.rst, self.rst, 3)
        self.feat_dims = feat_dims

        self.main_model = self.build_main_model()
        self.embedding = self.embedding_model()


    def feature_extractor(self):
        image = Input(self.input_shape)
        vgg16 = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=Input(self.input_shape),
            input_shape=self.input_shape,
        )

        x = vgg16(image)
        out1 = keras.layers.advanced_activations.PReLU(name='side_out')(x)
        out2 = Dense(self.num_of_classes, activation='softmax', name='main_out')(out1)
        return out1, out2
        

    def build_main_model(self):
        side_output, final_output = self.feature_extractor()
        centers = Embedding(num_of_classes, self.feat_dims)(labels)
        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),
                            name='l2_loss')([side_output ,centers])

        train_model = Model(inputs=[images, labels],
                            outputs=[final_output, l2_loss])
        train_model.compile(optimizer=Adam(self.lr),
                            loss=[
                                  "categorical_crossentropy",
                                  lambda y_true, y_pred: y_pred
                            ],
                            metrics=['accuracy'])
        return train_model


    def embbeding_model(self):
        return Model(inputs = main_model.inputs[0],
                     outputs = self.main_model.get_layer('side_out').get_output_at(-1),
                     name="center_loss")


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

            for x, y in data_gen.next_batch(augment_factor):
                loss, acc = self.main_model.train_on_batch(x, y)
                batch_loss['loss'].append(loss)

            # evaluation
            batch_loss['val_loss'] = self.main_model.evaluate(test_gen.x_test,
                                                              test_gen.y_test,
                                                              verbose=False)

            mean_loss = np.mean(np.array(batch_loss['loss']))
            mean_val_loss = np.mean(np.array(batch_loss['val_loss']))

            history['loss'].append(mean_loss)
            history['val_loss'].append(mean_val_loss)

            print("Loss: {}, Val Loss: {} - {}".format(
                mean_loss, mean_val_loss,
                datetime.datetime.now() - start_time
            ))

        self.history = history
        return history


    