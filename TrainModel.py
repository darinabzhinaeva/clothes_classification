import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from keras.callbacks import Callback, ModelCheckpoint
import keras.backend as K
import numpy as np
import gc

from CNNModel import get_model
from TrainDataGenerator import TrainDataGenerator

CHANNELS = 3
IMG_SIZE = (512, 512, CHANNELS)
EPOCHS = 80
TEST_SIZE = 1/6
RUN_NAME = 'Train1_'

best_val_loss = np.Inf
def CustomReduceLRonPlateau(model, history, epoch):
    global best_val_loss

    # ReduceLR Constants
    monitor = 'val_age_loss'
    patience = 5
    factor = 0.75
    min_lr = 1e-5

    # Get Current LR
    current_lr = float(K.get_value(model.optimizer.lr))

    # Print Current Learning Rate
    print('Current LR: {0}'.format(current_lr))

    # Monitor Best Value
    current_val_loss = history[monitor][-1]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
    print('Best Vall Loss: {0}'.format(best_val_loss))

    # Track last values
    if len(history[monitor]) >= patience:
        last5 = history[monitor][-5:]
        print('Last: {0}'.format(last5))
        best_in_last = min(last5)
        print('Min value in Last: {0}'.format(best_in_last))

        # Determine correction
        if best_val_loss < best_in_last:
            new_lr = current_lr * factor
            if new_lr < min_lr:
                new_lr = min_lr
            print('ReduceLRonPlateau setting learning rate to: {0}'.format(new_lr))
            K.set_value(model.optimizer.lr, new_lr)

# Modelcheckpoint
def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name,
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=True,
                           save_weights_only=True,
                           mode='min',
                           period=1)

def main():
    df = pd.read_csv("data.csv", sep=";")

    model = get_model(IMG_SIZE)

    model.compile(
        optimizer=Adam(lr=0.016),
        loss={"season": 'mean_absolute_error'},
        loss_weights={"season": 1},
        metric={"season": ["mean_absolute_error", tf.keras.metrics.Recall()]}
    )

    # Model summary
    print(model.summary())

    # Multi Label Stratified Split stuff...
    msss = MultilabelStratifiedShuffleSplit(n_splits=EPOCHS, test_size=TEST_SIZE)

    X_train = df["PATH"].values
    y_columns = [x for x in df.columns if x.startswith("SEASON")]
    Y_train = df[y_columns].to_numpy()

    for epoch, msss_split in zip(range(EPOCHS), msss.split(X_train, Y_train)):
        print('=========== EPOCH {}'.format(epoch))
        train_ids = msss_split[0]
        valid_ids = msss_split[1]

        print('Train Length: {0}   First 10 indices: {1}'.format(len(train_ids), train_ids[:]))
        print('Valid Length: {0}    First 10 indices: {1}'.format(len(valid_ids), valid_ids[:]))

        train_df = df.loc[train_ids]
        X_train_data = train_df["PATH"].values
        y_columns = [x for x in train_df.columns if x.startswith("SEASON")]
        Y_train_data = train_df[y_columns].to_numpy()

        data_generator_train = TrainDataGenerator(X_train_data,
                                                  Y_train_data,
                                                  train_ids,
                                                  batch_size=16,
                                                  img_size=IMG_SIZE)

        valid_df = df.loc[valid_ids]
        X_valid_data = valid_df["PATH"].values
        Y_valid_data = valid_df[y_columns].to_numpy()

        data_generator_val = TrainDataGenerator(X_valid_data,
                                                  Y_valid_data,
                                                  valid_ids,
                                                  batch_size=16,
                                                  img_size=IMG_SIZE)

        TRAIN_STEPS = int(len(data_generator_train))
        VALID_STEPS = int(len(data_generator_val))

        print('Train Generator Size: {0}'.format(len(data_generator_train)))
        print('Validation Generator Size: {0}'.format(len(data_generator_val)))

        model.fit_generator(
            generator=data_generator_train,
            validation_data=data_generator_val,
            steps_per_epoch=TRAIN_STEPS,
            validation_steps=VALID_STEPS,
            epochs=1,
            callbacks=[ModelCheckpointFull(RUN_NAME + 'model_' + str(epoch) + '.h5')],
            verbose=1)

        # Set and Concat Training History
        temp_history = model.history.history
        if epoch == 0:
            history = temp_history
        else:
            for k in temp_history: history[k] = history[k] + temp_history[k]

        # Custom ReduceLRonPlateau
        CustomReduceLRonPlateau(model, history, epoch)

        # Cleanup
        del data_generator_train, data_generator_val, train_ids, valid_ids
        gc.collect()

if __name__ == "__main__":
    main()