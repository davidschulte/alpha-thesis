import tensorflow.keras as keras
import numpy as np
import os
from utils import dotdict

args = dotdict({
    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 32,
    'num_channels': 256,
})


class NNetWrapper:

    def __init__(self, game):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.dropout = 0.5
        self.epochs = 5
        self.batch_size = 64
        self.num_channels = 256

        inputs = keras.Input(shape=(self.board_y, self.board_x))  # Returns a placeholder tensor

        action_size = game.getActionSize()

        x = keras.layers.Reshape((self.board_y, self.board_x, 1))(inputs)

        x = keras.layers.Conv2D(self.num_channels, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(self.num_channels, kernel_size=5, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(self.num_channels, kernel_size=3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(self.num_channels, kernel_size=5)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Reshape(((self.board_y-6)*(self.board_x-6)*self.num_channels,))(x)

        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)

        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(self.dropout)(x)

        pi = keras.layers.Dense(action_size, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.1), name='pi')(x)
        # pi = keras.layers.Softmax(name='pi')(pi)

        v = keras.layers.Dense(3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1), name='v')(x)
        # v = keras.layers.ReLU(name='v')(v)
        # v = keras.layers.Softmax()(v)
        # v = keras.layers.Lambda(lambda x: x * 3)(v)

        # x = tf.keras.layers.Dense(num_channels, activation='relu')(x)
        # x = tf.keras.layers.Dense(num_channels, activation='relu')(x)
        # predictions = tf.keras.layers.Dense(1, activation='sigmoid', name='pred1')(x)
        # second_predictions = tf.keras.layers.Dense(3, activation='sigmoid', name='pred2')(x)

        self.model = keras.Model(inputs=inputs, outputs=[pi, v])

        self.model.compile(optimizer='adam',
                      loss={'pi': 'categorical_crossentropy',
                          'v': 'mean_squared_error'},
                      metrics=['accuracy'])

    def train(self, examples):
        boards, pis, vs = list(zip(*examples))
        self.model.fit(np.array(boards), [np.array(pis), np.array(vs)], epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, board, player):
        board = board[np.newaxis, :, :]
        board = board.astype('float32')
        [pi, v] = self.model.predict(board)
        pi = np.reshape(pi, (self.action_size,))
        v = np.reshape(v, (3,))
        if player == 2:
            v = np.array([v[2], v[0], v[1]])
        elif player == 3:
            v = np.array([v[1], v[2], v[0]])

        return pi, v

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        self.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath+'.meta'):
        #     raise("No model in path {}".format(filepath))
        self.model = keras.models.load_model(filepath)
