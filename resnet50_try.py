from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout


def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. Need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer())(X)
    X = BatchNormalization(axis=3)(X, training=training)  # Default axis
    X = Activation('relu')(X)
    X = Dropout(rate=0.0)(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer())(X)
    X = BatchNormalization(axis=3)(X, training=training)  # Default axis
    X = Activation('relu')(X)
    X = Dropout(rate=0.0)(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer())(X)
    X = BatchNormalization(axis=3)(X, training=training)  # Default axis
    X = Dropout(rate=0.0)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = Dropout(rate=0.0)(X)

    return X


def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
    """
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer.

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path glorot_uniform
    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer())(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)
    X = Dropout(rate=0.0)(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer())(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)
    X = Dropout(rate=0.0)(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer())(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Dropout(rate=0.0)(X)

    # SHORTCUT PATH
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer())(
        X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)
    X = Dropout(rate=0.0)(X)

    # Final step: Add shortcut value to main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = Dropout(rate=0.0)(X)

    return X


def ResNet50(input_shape=IMG_SHAPE, classes=CLASSES_AMOUNT):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = ZeroPadding2D((3, 3))(X)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(ACTIVATION)(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = Dropout(rate=0.0)(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [64, 64, 256])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [64, 64, 256])
    X = Dropout(rate=0.0)(X)

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [128, 128, 512])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [128, 128, 512])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [128, 128, 512])
    X = Dropout(rate=0.0)(X)

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [256, 256, 1024])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [256, 256, 1024])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [256, 256, 1024])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [256, 256, 1024])
    X = Dropout(rate=0.0)(X)
    X = identity_block(X, 3, [256, 256, 1024])
    X = Dropout(rate=0.0)(X)

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = Dropout(rate=0.4)(X)
    X = identity_block(X, 3, [512, 512, 2048])
    X = Dropout(rate=0.4)(X)
    X = identity_block(X, 3, [512, 512, 2048])
    X = Dropout(rate=0.4)(X)

    # AVGPOOL
    X = AveragePooling2D((2, 2))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform())(X)

    # Create model
    model = Model(inputs=X_input, outputs=X)

    return model
