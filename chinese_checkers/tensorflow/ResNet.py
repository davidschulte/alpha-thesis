import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
from utils import dotdict


class NNetWrapper:

    def __init__(self, game):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.dropout = 0.3
        self.epochs = 5
        self.batch_size = 64
        self.num_channels = 128

        inputs = keras.Input(shape=(self.board_y, self.board_x))  # Returns a placeholder tensor

        x = keras.layers.Reshape((self.board_y, self.board_x, 1))(inputs)
        x = keras.layers.Conv2D(self.num_channels, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)
        x = self.res_block(x, self.num_channels)

        x = keras.layers.Reshape((self.board_y*self.board_x*self.num_channels,))(x)

        pi = self.policy_head(x, self.action_size)

        v = self.value_head(x)

        self.model = keras.Model(inputs=inputs, outputs=[pi, v])
        self.model.compile(optimizer='adam',
                      loss={'pi': 'categorical_crossentropy',
                          'v': 'mean_squared_error'},
                      metrics=['accuracy'])

    def res_block(self, input, filter_size):
        x = keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([input, x])
        x = keras.layers.ReLU()(x)
        return x

    def policy_head(self, input, action_size):
        x = keras.layers.Dense(256)(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(action_size)(x)
        pi = keras.layers.Softmax(name='pi')(x)
        return pi

    def value_head(self, input):
        x = keras.layers.Dense(256)(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(3)(x)
        v = keras.layers.ReLU(name='v')(x)
        return v

    def train(self, examples):
        boards, pis, vs = list(zip(*examples))
        self.model.fit(np.array(boards), [np.array(pis), np.array(vs)], epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, board):
        """
        predicts v and pi for one board state
        :param board: current board
        :return: predicitons
        """
        board = board[np.newaxis, :, :]
        board = board.astype('float32')
        [pi, v] = self.model.predict(board)
        pi = np.reshape(pi, (self.action_size,))
        v = np.reshape(v, (3,))
        return pi, v

    def predict_parallel(self, boards):
        """
        predicsts pis and vs for several board states
        :param boards: several board states combined in a list
        :return: predicitons
        """
        boards = tf.convert_to_tensor(boards, np.float32)
        return self.model.predict(boards)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        self.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        loads the model from a file
        :param folder:
        :param filename:
        """
        filepath = os.path.join(folder, filename)
        self.model = keras.models.load_model(filepath)

    def load_first_checkpoint(self, folder, iteration):
        """
        loads model from a file, only used when loading a model at the start of the program
        :param folder:          source folder
        :param iteration:       iteration number from model
        """
        filename = "checkpoint_" + str(iteration) + ".h5"
        filepath = os.path.join(folder, filename)
        self.model = keras.models.load_model(filepath)
