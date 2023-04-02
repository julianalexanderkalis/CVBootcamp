import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model
# Define a function that creates an encoding block for a U-Net model


def encBlock(inputs, filters, maxPooling=True):

    # Apply a 3x3 convolutional layer with 'filters' filters, followed by batch normalization and ReLU activation
    Layer = Conv2D(filters, 3, padding="same",
                   kernel_initializer="he_normal")(inputs)
    Layer = BatchNormalization()(Layer)
    Layer = Activation("relu")(Layer)

    # Apply another 3x3 convolutional layer with 'filters' filters, followed by batch normalization and ReLU activation
    Layer = Conv2D(filters, 3, padding="same",
                   kernel_initializer="he_normal")(Layer)
    Layer = BatchNormalization()(Layer)
    Layer = Activation("relu")(Layer)

    # Set aside the output of the second convolutional layer for use in the corresponding decoding block
    skipCon = Layer

    # If maxPooling is True, add a MaxPooling2D layer with 2x2 pool size to halve the spatial dimensions of the feature maps
    if maxPooling:
        next = MaxPooling2D(pool_size=(2, 2))(Layer)
    else:
        next = Layer

    return next, skipCon


def decBlock(inputs, skipConInput, filters):

    # Use Conv2DTranspose to upsample the inputs
    LayerTrans = Conv2DTranspose(filters, 3, strides=(
        2, 2), padding="same", kernel_initializer="he_normal")(inputs)

    # Concatenate the upsampled inputs with the skipConInput to form a residual connection
    resCon = concatenate([LayerTrans, skipConInput], axis=3)

    # Apply two sets of Conv2D -> BatchNormalization -> Activation layers
    Layer = Conv2D(filters, 3, padding="same",
                   kernel_initializer="he_normal")(resCon)
    Layer = BatchNormalization()(Layer)
    Layer = Activation("relu")(Layer)

    Layer = Conv2D(filters, 3, padding="same",
                   kernel_initializer="he_normal")(Layer)
    Layer = BatchNormalization()(Layer)
    Layer = Activation("relu")(Layer)

    return Layer


def defineUNetModel(inputSize, filters, outputNodes):

    inputs = Input(inputSize)

    # Encoding blocks
    E1, SC1 = encBlock(inputs, filters)
    E2, SC2 = encBlock(E1, filters * 2)
    E3, SC3 = encBlock(E2, filters * 4)
    E4, SC4 = encBlock(E3, filters * 8)

    # Bottleneck layer
    E5, _ = encBlock(E4, filters * 16, maxPooling=False)

    # Decoding blocks
    D6 = decBlock(E5, SC4, filters * 8)
    D7 = decBlock(D6, SC3, filters * 4)
    D8 = decBlock(D7, SC2, filters * 2)
    D9 = decBlock(D8, SC1, filters)

    # Output layers
    L10 = Conv2D(filters, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal')(D9)
    L11 = Conv2D(outputNodes, kernel_size=(1, 1),
                 activation='sigmoid', padding='same')(L10)
    # Define and return model
    model = Model(inputs=inputs, outputs=L11)

    return model

# Define a function to create masks for input images using the given model


def createMaskForImage(dataset, model):
    maskTrue = []
    maskPredE = []
    for img, m in dataset:
        # Use the model to predict the mask for the input image
        maskPred = model.predict(img)
        # Convert the predicted mask to a binary image
        maskPred = tf.expand_dims(tf.argmax(maskPred, axis=-1), axis=-1)
        # Append the ground truth mask and predicted mask to their respective lists
        maskTrue.extend(m)
        maskPredE.extend(maskPred)

    # Convert the lists of masks to NumPy arrays
    maskTrue = np.array(maskTrue)
    maskPredE = np.array(maskPredE)

    # Return the ground truth and predicted masks
    return maskTrue, maskPredE


def modelEvaluation(masksTrue, masksPred, n):

    # Initialize empty lists to store precision and recall values
    precision, recall = [], []
    truePositives, falsePositives, falseNegatives = 0, 0, 0

    for c in range(n):  # Iterate over each class label

        # Get the number of masks in the ground truth
        nMask = masksTrue.shape[0]

        for i in range(nMask):  # Iterate over each mask in the ground truth

            # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
            TP = np.sum(np.logical_and(masksTrue[i] == c, masksPred[i] == c))
            FP = np.sum(np.logical_and(masksTrue[i] != c, masksPred[i] == c))
            FN = np.sum(np.logical_and(masksTrue[i] == c, masksPred[i] != c))

            # Store the sum of TP, FP, TN, and FN in their respective variables
            truePositives += TP
            falsePositives += FP
            falseNegatives += FN

        # Calculate precision and recall for the current class and append them to the respective lists
        recall = round(truePositives/(truePositives + falsePositives), 2)
        precision = round(truePositives/(truePositives + falsePositives), 2)
        precision.append(precision)
        recall.append(recall)

    # Calculate the average precision and recall across all classes and return them as a dictionary
    return {"Overall Evaluations": {"Precision": round(np.average(np.array(precision)), 2), "Recall": round(np.average(np.array(recall)), 2)}}
