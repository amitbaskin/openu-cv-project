import tensorflow as tf
from tf.keras.applications.inception_v3 import InceptionV3
from tf.keras.models import Model
from tf.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten
from tf.keras.callbacks import ModelCheckpoint
from tf.keras.optimizers.schedules import ExponentialDecay
from tf.keras.optimizers import Adam
import matplotlib.pyplot as plt
from myconstants import *


class ModelTemplate:
    def __init__(self, base_model, training, validation,
                 img_shape=IMG_SHAPE, weights=WEIGHTS, classes_amount=CLASSES_AMOUNT):
        self.base_model = base_model
        self.training = training
        self.validation = validation
        self.img_shape = img_shape
        self.weights = weights
        self.classes_amount = classes_amount


def get_inception_template(training, validation):
    return ModelTemplate(InceptionV3, training, validation)


def get_freezed_learning_schedule():
    return ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.97,
            staircase=True)


def get_freezed_optimizer():
    return Adam(learning_rate=get_freezed_learning_schedule())


def get_unfreezed_optimizer():
    return Adam(learning_rate=get_unfreezed_learning_schedule())


def get_unfreezed_learning_schedule():
    return ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=10000,
            decay_rate=0.97,
            staircase=True)


def get_callback_checkpoint(model_path):
    return ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch')


def plot_hist(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


class InceptionModel:
    def __init__(self, model_path, training, validation):
        self.model_path = model_path
        self.checkpoint = get_callback_checkpoint(self.model_path)
        self.template = get_inception_template(training, validation)
        mymodel = self.template.base_model(InceptionV3, IMG_SHAPE, weights)
        input_tensor = Input(shape=self.template.img_shape)
        base_model = mymodel.base_model(input_tensor=input_tensor, weights=self.template.weights, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D(name="avg_pool")(x)
        x = Flatten(name="flat")(x)
        x = Dense(512, activation='relu', name="dense")(x)
        x = BatchNormalization(name="norm")(x)
        x = Dropout(0.5, name="top_dropout")(x)
        predictions = Dense(self.template.classes_amount, activation='softmax', name="pred")(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

    def freeze(self):
        for layer in self.template.base_model.layers:
            layer.trainable = False

    def fit_freezed(self):
        return self.model.fit(
            self.template.training, epochs=5, validation_data=self.template.validation, callbacks=[self.checkpoint])

    def fit_unfreezed(self):
        return self.model.fit(
            self.template.training, epochs=10, validation_data=self.template.validation, callbacks=[self.checkpoint])

    def unfreeze(self):
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True

    def train_freezed(self):
        model.compile(optimizer=get_freezed_optimizer(), loss='categorical_crossentropy', metrics=["accuracy"])
        history = self.fit_freezed()
        plot_hist(history)

    def train_unfreezed(self):
        self.model.compile(optimizer=get_unfreezed_optimizer(), loss='categorical_crossentropy', metrics=["accuracy"])
        self.model.load_weights(self.model_path)
        history = self.fit_unfreezed()
        plot_hist(history)

    def train(self):
        self.freeze()
        self.train_freezed()
        self.train_unfreezed()
