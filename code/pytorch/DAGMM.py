from multiprocessing import Manager
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Flatten, Dense, Conv2DTranspose, Reshape
from utils import get_channels_axis
import torch
import torch.nn as nn

def conv_encoder(input_side=32, n_channels=3, representation_dim=256, representation_activation='tanh',
                 intermediate_activation='relu'):
    nf = 64
    input_shape = (n_channels, input_side, input_side) if get_channels_axis() == 1 else (input_side, input_side,
                                                                                         n_channels)

    x_in = Input(shape=input_shape)
    enc = x_in

    # downsample x0.5
    enc = Conv2D(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(nf * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(nf * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
    enc = BatchNormalization(axis=get_channels_axis())(enc)
    enc = Activation(intermediate_activation)(enc)

    if input_side == 64:
        # downsample x0.5
        enc = Conv2D(nf * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(enc)
        enc = BatchNormalization(axis=get_channels_axis())(enc)
        enc = Activation(intermediate_activation)(enc)

    enc = Flatten()(enc)
    rep = Dense(representation_dim, activation=representation_activation)(enc)

    return Model(x_in, rep)


def conv_decoder(output_side=32, n_channels=3, representation_dim=256, activation='relu'):
    nf = 64

    rep_in = Input(shape=(representation_dim,))

    g = Dense(nf * 4 * 4 * 4)(rep_in)
    g = BatchNormalization(axis=-1)(g)
    g = Activation(activation)(g)

    conv_shape = (nf * 4, 4, 4) if get_channels_axis() == 1 else (4, 4, nf * 4)
    g = Reshape(conv_shape)(g)

    # upsample x2
    g = Conv2DTranspose(nf * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = BatchNormalization(axis=get_channels_axis())(g)
    g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = BatchNormalization(axis=get_channels_axis())(g)
    g = Activation(activation)(g)

    if output_side == 64:
        # upsample x2
        g = Conv2DTranspose(nf, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
        g = BatchNormalization(axis=get_channels_axis())(g)
        g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(n_channels, kernel_size=(3, 3), strides=(2, 2), padding='same')(g)
    g = Activation('tanh')(g)

    return Model(rep_in, g)

def _dagmm_experiment(selected_label, gpu_q, p, cycle):
    gpu_to_use = gpu_q.get()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

    n_channels = 1
    input_side = x_train.shape[2]  # image side will always be at shape[2]
    enc = conv_encoder(input_side, n_channels, representation_dim=5,
                       representation_activation='linear')
    dec = conv_decoder(input_side, n_channels=n_channels, representation_dim=enc.output_shape[-1])
    n_components = 3
    estimation = Sequential([Dense(64, activation='tanh', input_dim=enc.output_shape[-1] + 2), Dropout(0.5),
                             Dense(10, activation='tanh'), Dropout(0.5),
                             Dense(n_components, activation='softmax')]
                            )

    batch_size = 1024
    epochs = 200
    lambda_diag = 0.005
    lambda_energy = 0.1
    dagmm_mdl = dagmm.create_dagmm_model(enc, dec, estimation, lambda_diag)
    optimizer = keras.optimizers.Adam(lr=1e-4)  # default config
    dagmm_mdl.compile(optimizer, ['mse', lambda y_true, y_pred: lambda_energy*y_pred])

    x_train_task = x_train
    x_test_task = x_train  # This is just for visual monitoring
    dagmm_mdl.fit(x=x_train_task, y=[x_train_task, np.zeros((len(x_train_task), 1))],  # second y is dummy
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test_task, [x_test_task, np.zeros((len(x_test_task), 1))]),
                  # verbose=0
                  )

    energy_mdl = Model(dagmm_mdl.input, dagmm_mdl.output[-1])

    scores = -energy_mdl.predict(x_train, batch_size)
    scores = scores.flatten()
    if not np.all(np.isfinite(scores)):
        min_finite = np.min(scores[np.isfinite(scores)])
        scores[~np.isfinite(scores)] = min_finite - 1
    labels = y_train.flatten()

    gpu_q.put(gpu_to_use)
    
if __name__ == '__main__':
    N_GPUS = 1
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))
    
    for label in range(0, 10):
        for cycle in range(0, 5):
            _dagmm_experiment(label,0.25,q,1,cycle=cycle)