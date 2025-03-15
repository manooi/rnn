import tensorflow as tf

class RNNModel:
    """
    A class to define, compile, and train the Simple RNN model.
    """
    @staticmethod
    def build_model(history_shape, future_target, node):
        """
        Builds a Simple RNN model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.SimpleRNN(node, input_shape=history_shape, activation='relu'))
        model.add(tf.keras.layers.Dense(future_target))
        return model

    @staticmethod
    def compile_and_fit(model, train_data, val_data, steps_per_epoch, epochs, validation_steps, learning_rate):
        """
        Compiles and trains the RNN model.
        """
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss='mae', metrics=['mae'])
        history = model.fit(train_data,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_data,
                            validation_steps=validation_steps)
        return history