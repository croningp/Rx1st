from itertools import permutations
from math import factorial
from os import path

import fast_jtnn
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from fast_jtnn.fp_calculator import FingerprintCalculator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

JTNN_HOME = path.dirname(fast_jtnn.__path__[0])
JTNN_MODEL_PATH = path.join(JTNN_HOME, "data", "models", "model.iter-20000")
JTNN_MOLS = path.join(JTNN_HOME, "data", "full_train.txt")
JTNN_VOCAB_PATH = path.join(JTNN_HOME, "data", "vocab_full.txt")


class Explorer:
    def __init__(self):

        with open(JTNN_MOLS, "r") as f:
            smiles = [l.strip() for l in f]
        fp_calculator = FingerprintCalculator(
            model_path=JTNN_MODEL_PATH, vocab_path=JTNN_VOCAB_PATH
        )
        fps = fp_calculator(smiles[:1000])

        self.normalizer = tfk.layers.experimental.preprocessing.Normalization()
        self.normalizer.adapt(fps)

    def nn_model(
        self,
        normalizer,
        dropout0=0.4,
        dropout1=0.2,
        dropout2=0.0,
        dropout3=0.0,
        fp_length=56,
    ):
        np.random.seed(2018)  # Happy new year :)

        fp_input = tfk.Input(shape=(3, fp_length))
        #     fp_present = tfk.Input(shape=(3,))
        rx_input = tfk.Input(shape=(5,))

        fp_processing = tfk.Sequential(
            [
                normalizer,
                tfk.layers.Dropout(dropout0),
                tfk.layers.Dense(32, activation="relu"),
                tfk.layers.Dropout(dropout0),
                tfk.layers.Dense(16, activation="relu"),
                tfk.layers.Dropout(dropout1),
            ]
        )
        fps = tf.stack([fp_processing(fp_input[:, i, :]) for i in range(3)], axis=1)
        fps = tfk.layers.Flatten()(fps)

        rx = tfk.layers.Dense(3, activation="relu")(rx_input)

        result = tfk.layers.concatenate(
            [
                fps,
                rx,
                #         fp_present
            ]
        )
        result = tfk.layers.Dense(24, activation="relu")(result)
        result = tfk.layers.Dropout(dropout2)(result)
        result = tfk.layers.Dense(16, activation="relu")(result)
        result = tfk.layers.Dropout(dropout3)(result)
        result = tfk.layers.Dense(8, activation="relu")(result)
        result = tfk.layers.Dense(4, activation="relu")(result)
        result = tfk.layers.Dense(1, activation="sigmoid")(result)

        # return conv1, conv2, pool1, pool2, input1, input2, input, comb, result
        mdl = tfk.Model(
            inputs=[
                fp_input,
                #             fp_present,
                rx_input,
            ],
            outputs=result,
        )

        mdl.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mean_absolute_error"],
        )

        return mdl

    def one_hot_model(self, dropout1=0.075, dropout2=0.1, dropout3=0.0, fp_length=56):
        np.random.seed(2018)  # Happy new year :)

        fp_input = tfk.Input(shape=(fp_length,))

        result = tfk.layers.Dense(32, activation="relu")(fp_input)
        result = tfk.layers.Dropout(dropout1)(result)

        result = tfk.layers.Dense(24, activation="relu")(result)
        result = tfk.layers.Dropout(dropout2)(result)
        result = tfk.layers.Dense(16, activation="relu")(result)
        result = tfk.layers.Dropout(dropout3)(result)
        result = tfk.layers.Dense(8, activation="relu")(result)
        result = tfk.layers.Dense(4, activation="relu")(result)

        result = tfk.layers.Dense(1)(result)

        # return conv1, conv2, pool1, pool2, input1, input2, input, comb, result
        mdl = tfk.Model(inputs=fp_input, outputs=result)

        mdl.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["mean_absolute_error"],
        )

        return mdl

    def train(
        self,
        reagent_fps,
        reactant_mat,
        predictions,
        sample_weights,
        normalizer=None,
        fit_kwargs={},
    ):
        normalizer = normalizer or self.normalizer
        (
            reagent_train,
            reagent_test,
            reactant_train,
            reactant_test,
            pred_train,
            pred_test,
            weights_train,
            weights_test,
        ) = train_test_split(
            reagent_fps,
            reactant_mat,
            predictions,
            sample_weights,
            test_size=0.2,
            random_state=2018,
        )

        self.reagent_train = reagent_train
        self.reactant_train = reactant_train
        self.pred_train = pred_train
        self.weights_train = weights_train

        aug_factor = factorial(3)
        (
            self.reagent_train_aug,
            self.reactant_train_aug,
            self.pred_train_aug,
            self.weights_train_aug,
        ) = [
            np.zeros(shape=(m.shape[0] * aug_factor,) + m.shape[1:], dtype=m.dtype)
            for m in [
                self.reagent_train,
                self.reactant_train,
                self.pred_train,
                self.weights_train,
            ]
        ]
        for i, p in enumerate(self.pred_train):
            for j, perm in enumerate(permutations(range(3))):
                ind = i * aug_factor + j
                self.reagent_train_aug[ind] = self.reagent_train[i, perm, :]
                self.reactant_train_aug[ind] = self.reactant_train[i]
                self.pred_train_aug[ind] = self.pred_train[i]
                self.weights_train_aug[ind] = self.weights_train[i]

        early_stopping = EarlyStopping(
            patience=50, verbose=1, restore_best_weights=True
        )
        self.model = self.nn_model(normalizer=self.normalizer)
        self.model.fit(
            x=[self.reagent_train_aug, self.reactant_train_aug],
            y=self.pred_train_aug,
            sample_weight=self.weights_train_aug,
            validation_data=([reagent_test, reactant_test], pred_test, weights_test),
            callbacks=[early_stopping],
            epochs=1000,
            verbose=0,
            **fit_kwargs
        )

    def train_one_hot(
        self, reagent_vec, predictions, sample_weights, model_kwargs={}, fit_kwargs={}
    ):
        (
            reagent_train,
            reagent_test,
            pred_train,
            pred_test,
            weights_train,
            weights_test,
        ) = train_test_split(
            reagent_vec, predictions, sample_weights, test_size=0.2, random_state=2018
        )

        self.reagent_train = reagent_train
        self.pred_train = pred_train
        self.weights_train = weights_train

        early_stopping = EarlyStopping(
            patience=100, verbose=1, restore_best_weights=True
        )
        self.model = self.one_hot_model(
            fp_length=reagent_train.shape[1], **model_kwargs
        )
        self.model.fit(
            x=self.reagent_train,
            y=self.pred_train,
            sample_weight=self.weights_train,
            validation_data=(reagent_test, pred_test, weights_test),
            callbacks=[early_stopping],
            epochs=1000,
            verbose=1,
            **fit_kwargs
        )

    def test(self, reagent_test, reactant_test):
        return self.model([reagent_test, reactant_test]).numpy()[:, 0]
