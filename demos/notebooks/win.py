import caiman as cm
import cv2
from caiman.utils.utils import download_demo
from time import time, sleep


def append_to_queue(q, init_batch, T, fr):
    t_start = time()
    for t in range(init_batch, T):
        # read frame and append to queue
        frame = next(iterator)
        q.put(frame)
        sleep(max(0, (t+1-init_batch)/fr - time() + t_start))


def get_iterator(device=0, fr=None):
    """
    device: device number (int) or filename (string) for reading from camera or file respectively
    fr: frame rate
    """
    if isinstance(device, int):  # capture from camera
        def capture_iter(device=device, fr=fr):
            cap = cv2.VideoCapture(device)
            if fr is not None:  # set frame rate
                cap.set(cv2.CAP_PROP_FPS, fr)
            while True:
                yield cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        iterator = capture_iter(device, fr)
    else:  # read frame by frame from file
        iterator = cm.base.movies.load_iter(device, var_name_hdf5='Y')
    return iterator


init_batch = 500  # number of frames to use for initialization

iterator = get_iterator(download_demo('blood_vessel_10Hz.mat'))
for t in range(init_batch):
    next(iterator)
