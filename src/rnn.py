from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
import pickle
from keras.models import model_from_json


def multivariate_data(dataset, target, history_size, target_size):
  data = []
  labels = []
  start_index = 0 + history_size
  end_index = len(dataset) - target_size
  for i in range(start_index, end_index):
    indices = range(i-history_size, i, 1)
    data.append(dataset[indices])
    labels.append(target[i+target_size])

  return np.array(data), np.array(labels)


def create_model(num_layers, num_units, num_dropout, shape):
    # start the model making process and create our first layer
    model = Sequential()
    if num_layers > 1:
        model.add(GRU(num_units, input_shape=shape,
                      recurrent_dropout=num_dropout, return_sequences=True))
    # create a loop making a new layer for the amount passed to this model.
    if num_layers > 2:
        for i in range(num_layers - 2):
            model.add(GRU(num_units, recurrent_dropout=num_dropout,
                          return_sequences=True))
        model.add(GRU(num_units, recurrent_dropout=num_dropout))
    else:
        model.add(GRU(num_units, input_shape=shape,
                      recurrent_dropout=num_dropout))
    model.add(Dense(1))
    # setup our optimizer and compile
    adam = Adam()
    model.compile(optimizer=adam, loss='mse')
    return model



class RNN:

    def __init__(self, x_train, y_train, x_val, y_val, x, y, data):
        self.model_variables = x_train.columns.tolist()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x = x
        self.y = y
        self.data = data
        self.data_mean = x_train.values.mean(axis=0)
        self.data_std = x_train.values.std(axis=0)
        self.x_train_scaled = (x_train.values - self.data_mean) / self.data_std
        self.x_val_scaled = (x_val.values - self.data_mean) / self.data_std
        self.x_scaled = (x.values - self.data_mean) / self.data_std

    def hyperpar_optimization(self, model_run_name):

        dim_timesteps = Integer(low=8, high=9, name='timesteps')
        dim_learning_rate = Real(low=1e-4, high=1e-1, prior='log-uniform',
                                 name='learning_rate')
        dim_num_layers = Integer(low=1, high=3, name='num_layers')
        dim_num_units = Integer(low=1, high=128, name='num_units')
        dim_num_dropout = Real(low=0, high=0.5, name='num_dropout')
        dim_batch_size = Integer(low=1, high=128, name='batch_size')
        dim_adam_decay = Real(low=1e-6, high=1e-2, name="adam_decay")

        dimensions = [dim_timesteps,
                      dim_num_layers,
                      dim_num_units,
                      dim_num_dropout,
                      dim_batch_size
                      ]
        default_parameters = [[8, 1, 10, 0, 20],
                              [8, 1, 64, 0.2, 20],
                              [8, 3, 64, 0.4, 10]]

        @use_named_args(dimensions=dimensions)
        def fitness(timesteps, num_layers, num_units, num_dropout,
                    batch_size):
            print(f"ts:{timesteps}, nlay:{num_layers}, nunits:{num_units}, drop:{num_dropout}, bs:{batch_size}")
            # model
            shape = x_train_single.shape[-2:]
            tf.compat.v1.set_random_seed(42)
            model = create_model(
                num_layers=num_layers,
                num_units=num_units,
                num_dropout=num_dropout,
                shape=shape
            )
            es = EarlyStopping(monitor='val_loss', verbose=0, patience=3,
                               restore_best_weights=True)
            # train model
            model_fit = model.fit(x=x_train_single,
                                  y=y_train_single,
                                  epochs=100,
                                  batch_size=batch_size,
                                  validation_data=(x_val_single, y_val_single),
                                  callbacks=[es], verbose=0
                                  )
            # return the validation mse
            mse = np.min(model_fit.history['val_loss'])
            # Print the classification accuracy.
            print("validation RMSE: {0:.5}".format(np.sqrt(mse)))
            # Delete the Keras model with these hyper-parameters from memory.
            del model
            # Clear the Keras session, otherwise it will keep adding new
            # models to the same TensorFlow graph each time we create
            # a model with a different set of hyper-parameters.
            tf.keras.backend.clear_session()
            tf.reset_default_graph()
            return mse

        np.random.RandomState(seed=42)
        gp_result = gp_minimize(func=fitness,
                               dimensions=dimensions,
                               n_calls=60,
                               n_jobs=-1,
                               kappa=5,
                               x0=default_parameters)
        with open("results" + model_run_name + ".txt", 'wb') as fp:
            pickle.dump(gp_result, fp)
        print(f"Best validation RMSE was {round(np.sqrt(gp_result.fun), 4)}")

