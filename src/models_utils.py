import tensorflow as tf
import tensorflow_io as tfio

def convert_to_keras_input(df):
    """Converts a dataframe into a dictionary of numpy arrays for Keras."""
    # Input features
    X_dict = {col: df[col].values for col in df.columns if col.startswith("x_")}
    z_dict = {col: df[col].values for col in df.columns if col == "z"}
    # Output/target variable
    y_imp = df["intervention"].values
    y_clk = df["outcome"].values
    return X_dict, z_dict, y_imp, y_clk
    

def naive(X_train, learning_rate=0.001):
    inputs = {key: tf.keras.layers.Input(shape=(1,), name=key) for key in X_train.keys}

    input_num = tf.keras.layers.Concatenate(axis=-1)(list(inputs.values()))
    input_num = tf.keras.layers.BatchNormalization()(input_num)

    pCTR = tf.keras.layers.Dense(256, activation="swish")(input_num)
    pCTR = tf.keras.layers.BatchNormalization()(pCTR)
    pCTR = tf.keras.layers.Dense(256, activation="relu")(pCTR)
    pCTR = tf.keras.layers.BatchNormalization()(pCTR)
    pCTR = tf.keras.layers.Dense(256, activation="relu")(pCTR)
    pCTR = tf.keras.layers.BatchNormalization()(pCTR)
    pCTR_output = tf.keras.layers.Dense(1, activation="sigmoid", name="click")(pCTR)

    model = tf.keras.Model(inputs=inputs, outputs=pCTR_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )

    return model
