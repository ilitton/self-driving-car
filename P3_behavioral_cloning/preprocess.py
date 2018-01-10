import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.image as mpimg

def resize_imgs(img, resize_dims=(66,200)):
    """Resize each image to the specified dimensions 
    
    Parameters
    ----------
    img : numpy array
        Original image represented as an array
    
    resize_dims : tuple of ints
        Dimensions for new image
    
    Returns
    -------
    resized_img : 3-D float Tensor of shape [new_height, new_width, channels]
        Image with new dimensions
    """
    import tensorflow as tf
    return tf.image.resize_images(img, (resize_dims))

def preprocess_img(img, reshape_dims=(200,66)):
    """Helper function to transform a single image to gray scale and normalize to force the range of pixel intensity to [0,1]
    
    Parameters
    ----------
    img : numpy array
        Original image represeted as an array
    
    Returns
    -------
    new_img : numpy array
        Image with one color channel and with a range of pixel intensity of [0,1] 
    """
    new_img = img[60:140,40:280]
    # Convert to gray 
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    new_img = cv2.equalizeHist(new_img)
    # Create a CLAHE obj and apply
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    new_img = clahe.apply(new_img)
    # Apply normalization
    new_img = new_img/127.5 - 1.
    # Reshape img
    new_img = cv2.resize(new_img, reshape_dims, interpolation = cv2.INTER_AREA)
    return new_img

def generator(samples, batch_size=32, correction=0.25):
    """Create generattor to load img data and process on the fly
    
    Parameters
    ----------
    samples : list of lists
        List of [center image filename, left image filename, right filename, steering, throttle, break, speed]
    
    batch_size : int
        Size of batch
    
    Returns
    -------
    X_train, y_train : tuple where
        X_train : list of shuffled training images
        y_train : list of shuffled steering wheel angles
    """
    num_samples = len(samples)
    # Loop forever so the generator neveer terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + correction
                right_angle = float(batch_sample[3]) - correction
                
                augmented_center_angle = center_angle * -1.0
                augmented_left_angle = (left_angle * -1.0) - correction
                augmented_right_angle = (right_angle * -1.0) + correction
                
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    # img = preprocess_img(cv2.imread(name))
                    img = mpimg.imread(name)
                    images.append(img)
                    images.append(cv2.flip(img, 1))
                
                angles += [center_angle, augmented_center_angle, left_angle, augmented_left_angle, right_angle, augmented_right_angle]
                # angles += [center_angle, left_angle, right_angle]
                
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)