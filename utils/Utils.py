from collections import Counter
from time import time

from tensorflow import keras


def get_generators(target_size: tuple = (135, 180), batch_size: int = 32) -> tuple:
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory('./images/train', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
    val_generator = train_datagen.flow_from_directory('./images/val', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
    test_generator = train_datagen.flow_from_directory('./images/test', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
    return train_generator, val_generator, test_generator

def scheduler(epoch):
    if epoch < 200:
        return .001
    if epoch < 400:
        return .0005

    return .0001


def top_k(l: list, k=2) -> list:
    """The counter.most_common([k]) method works
    in the following way:
    >>> Counter('abracadabra').most_common(3)  
    [('a', 5), ('r', 2), ('b', 2)]
    """

    c = Counter(l)
    return [key for key, val in c.most_common(k)]


def hasAmplifier(l: list) -> tuple:
    """Search for an element that has amplifier in it's name

    Arguments:
        l {list} -- elements haystakck

    Returns:
        tuple -- amplifierFounded => bool, ordered actions => list
    """
    ret = []
    amplifier_found = False
    for element in l:
        if 'actionAmplifier' in element:
            ret.insert(0, element)
            amplifier_found = True
        else:
            ret.append(element)

    return amplifier_found, ret

def getFrames(cam, s=5):
    """Get all the frames of the cam capture within the number of seconds given

    Arguments:
        cam {VideoCapture} -- the camera that captures the video

    Keyword Arguments:
        s {int} -- The number of seconds (default: {5})

    Yields:
        generator -- every frame
    """
    start = time()
    while (time() - start) < s: # take frames for S seconds
        _, frame = cam.read()
        yield frame
