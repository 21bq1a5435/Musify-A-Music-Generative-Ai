import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import os

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """
    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0], return_sequences=False)(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Train and save TF model with checkpointing.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # Check if a saved model exists
    if os.path.exists(SAVE_MODEL_PATH):
        print("Loading saved model...")
        model = keras.models.load_model(SAVE_MODEL_PATH)
    else:
        print("Building a new model...")
        model = build_model(output_units, num_units, loss, learning_rate)

    # add checkpointing
    checkpoint = keras.callbacks.ModelCheckpoint(
        SAVE_MODEL_PATH, 
        monitor="loss", 
        save_best_only=True, 
        verbose=1
    )

    # train the model
    model.fit(inputs, targets, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=[checkpoint])

if __name__ == "__main__":
    train()
