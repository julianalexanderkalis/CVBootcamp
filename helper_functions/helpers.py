import os
import tensorflow as tf

# Define a function that takes a list of directory paths as its input parameter


def createImagePaths(directoryPaths):
    # Create an empty list to store the image paths
    imagePaths = []

    # Loop through each directory in the list of directory paths
    for directory in range(len(directoryPaths)):
        # Get the list of file names in the current directory
        names = os.listdir(directoryPaths[directory])

        # Loop through each file name in the current directory
        for n in names:
            # Append the full path of the current file to the list of image paths
            imagePaths.append(directoryPaths[directory] + n)

    # Return the list of image paths
    return imagePaths


# Define a function that reads an image and its corresponding mask from file paths
def readImage(imagePath, maskPath):
    # Read the image file from the given file path
    image = tf.io.read_file(imagePath)

    # Decode the image from PNG format to a 3-channel tensor
    image = tf.image.decode_png(image, channels=3)

    # Convert the image tensor to float32 data type and normalize its pixel values
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image tensor to a fixed size using the nearest-neighbor method
    image = tf.image.resize(image, (256, 256), method='nearest')

    # Read the mask file from the given file path
    mask = tf.io.read_file(maskPath)

    # Decode the mask from PNG format to a 3-channel tensor
    mask = tf.image.decode_png(mask, channels=3)

    # Compute the maximum value across the color channels for each pixel in the mask
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

    # Resize the mask tensor to a fixed size using the nearest-neighbor method
    mask = tf.image.resize(mask, (256, 256), method='nearest')

    # Return the image and mask tensors as a tuple
    return image, mask


# Define a function that creates a TensorFlow dataset from a list of image and mask paths
def dataGenerator(imagePaths, maskPaths, bufferSize, batchSize):

    # Convert the input lists to TensorFlow constants
    imageList = tf.constant(imagePaths)
    maskList = tf.constant(maskPaths)

    # Create a TensorFlow dataset from the constants
    dataset = tf.data.Dataset.from_tensor_slices((imageList, maskList))

    # Apply the read_image function to each element of the dataset in parallel
    dataset = dataset.map(readImage, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache the dataset in memory for faster training, shuffle it, and batch it
    dataset = dataset.cache().shuffle(bufferSize).batch(batchSize)

    # Return the resulting dataset
    return dataset
