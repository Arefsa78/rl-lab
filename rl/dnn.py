from keras import layers, Sequential, Model
from keras.models import load_model
import keras.losses

from rl.data import Data

import numpy as np


class Dnn:
    VERBOSE = False
    def __init__(self, input_shape: int=0, output_shape: int=0, hidden_layer_size: list=[], activisions: list=[], name="MyDnn", load_dir:str = "") -> None:
        if len(load_dir) > 0:
            self.inputs = None
            self.hidden_layer = None
            self.outputs = None
            self.model = load_model(load_dir)

            return
        
        self.inputs = layers.Input(shape=(input_shape, ))

        self.hidden_layer = None
        for l in range(len(hidden_layer_size)):
            if self.hidden_layer is None:
                self.hidden_layer = layers.Dense(
                    hidden_layer_size[l], activisions[l])(self.inputs)
            else:
                self.hidden_layer = layers.Dense(
                    hidden_layer_size[l], activisions[l])(self.hidden_layer)
        
        self.outputs = layers.Dense(output_shape, "linear")(self.hidden_layer)
        self.model = Model(inputs=self.inputs, outputs=self.outputs, name=name)
    
    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer="adam",
            metrics=["categorical_accuracy"],
        )

    def learn(self, batch_x, batch_y, epochs):
        return self.model.fit(batch_x, batch_y, epochs=epochs, use_multiprocessing=True,verbose=Dnn.VERBOSE)

    def predict(self, x):
        x = x.reshape((1, 8))
        return self.model.predict(x, use_multiprocessing=True, verbose=Dnn.VERBOSE)
    
    def save(self, name):
        self.model.save(name)

    def get_weights(self):
        return self.model.get_weights()