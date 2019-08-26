import sys

import caffe2
import cv2
import matplotlib.pyplot as plt


# import Image

def deploy(img_path):

    # utilize the GPU if one is available
    # TODO replace with model.cuda() for PyTorch
    caffe2.set_mode_gpu()

    # directories containing the dogs and cats dataset and the
    # corresponding files for the model trained on the dataset
    MODEL_JOB_DIR = '/dli/data/digits/20180301-185638-e918'
    DATASET_JOB_DIR = '/dli/data/digits/20180222-165843-ada0'

    # the model architecture and weights files (from DIGITS)
    ARCHITECTURE = MODEL_JOB_DIR + '/deploy.prototxt'
    WEIGHTS = MODEL_JOB_DIR + '/snapshot_iter_735.caffemodel'

    # initialize the Caffe model using the model trained in DIGITS
    net = caffe.Classifier(ARCHITECTURE, WEIGHTS,
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(256, 256))

    # create an input that the network expects (i.e. a resolution
    # of 256 x 256 and with the mean image subtracted)
    input_image = caffe.io.load_image(img_path)
    test_image = cv2.resize(input_image, (256, 256))
    mean_image = caffe.io.load_image(DATASET_JOB_DIR + '/mean.jpg')
    test_image = test_image - mean_image

    # utilize the model for inference (predict cat or dog)
    prediction = net.predict([test_image])

    # display the input image
    print("Input Image:")
    plt.imshow(sys.argv[1])
    plt.show()

    # Image.open(input_image).show()

    print("Prediciton:")
    print(prediction)

    # create a useful output
    print("Output:")
    if prediction.argmax() == 0:
        print("Sorry cat:( https://media.giphy.com/media/jb8aFEQk3tADS/giphy.gif")
    else:
        print("Welcome dog! https://www.flickr.com/photos/aidras/5379402670")


##Ignore this part
if __name__ == '__main__':

    #
    image_path = sys.argv[1]
    print(deploy(image_path))
