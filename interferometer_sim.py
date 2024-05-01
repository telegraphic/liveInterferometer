"""
# Interferometer simulator, using webcam

Based on 'liveinterferometer' by Jack Hickish, Griffin Foster, and Charles Copley
https://github.com/jack-h/liveInterferometer

"""
import cv2
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


##############
## OpenCV camera + imaging methods
##############

def test_camera(cam: cv2.VideoCapture):
    """ Tests the camera is working by capturing an image

    Args:
        cam (cv2.VideoCapture): Camera device to test
    """
    rv, img = cam.read()
    if img is None or rv is False:
        raise RuntimeError("Camera not working")

def get_aspect_ratio(cam: cv2.VideoCapture) -> float:
    """ Returns the aspect ratio for the camera

     Args:
        cam (cv2.VideoCapture): Camera device to query

    Returns:
         aspect_ratio (float): Aspect ratio of camera frames
    """
    rv, img = cam.read()
    return img.shape[1] / img.shape[0]

def read_from_camera(cam: cv2.VideoCapture, xsize: int, ysize: int, greyscale: bool=False) -> np.ndarray:
    """ Read a frame from the camera

     Args:
        cam (cv2.VideoCapture): Camera device
        xsize (int): Image output width, in pixels
        ysize (int): Image output height, in pixels
        greyscale (bool): Converts image to greyscale if set to True.

    Returns:
         img (np.ndarray): Image as a numpy array.
                           If grayscale, image has shape (ysize, xsize)
                           If color (RGB), image has shape (ysize, xsize, 3)
    """
    # To avoid buffering issues, we read a few frames at once
    for i in range(4):
        rv, img = cam.read()
    img = cv2.resize(img, (xsize, ysize), interpolation=cv2.INTER_CUBIC)
    if greyscale:
        img = to_greyscale(img)
    else:
        img = to_rgb(img)
    img = np.fliplr(img)
    return img

def to_greyscale(img: np.ndarray) -> np.ndarray:
    """ Converts an image to greyscale, using CV2"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_rgb(img: np.ndarray) -> np.ndarray:
    """ Converts a BGR (webcam default) image to RGB (computer default) """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_image(filepath: str, xsize: int=None, ysize: int=None, greyscale: bool=False) -> np.ndarray:
    """ Loads target image using CV2

    Args:
        filepath (str): Path to file to load
        xsize (int): If set, will resize image to be xsize pixel wide.
                     Note: ysize must be set too.
        ysize (int): If set, will resize image to be ysize pixels high.
                     Note: xsize must be set too.
        greyscale (int): Convert to greyscale if True (default False)

    Returns:
        img (np.ndarray): Image as a numpy array
    """
    try:
        img = cv2.imread(filepath)
        if xsize is not None:
            img = cv2.resize(img, (xsize, ysize), interpolation=cv2.INTER_CUBIC)
        if greyscale:
            img = to_greyscale(img)
            #img[img < 100] = 0
        else:
            img = to_rgb(img)
    except cv2.error:
        raise ValueError(f"Could not load {filepath}")
    return img

def detect_stations(img: np.ndarray, xsize: int, ysize: int, rescale_factor: int=4) -> tuple:
        """ Detect station locations in an image using a Hough Circle detection algorithm

        Station locations are represented in the image by circles (intended for coins)
        Note that the circle detection is a bit flaky, and may need some hand tuning.

        Args:
            img (np.ndarray): Input image with circles corresponding to station locations
            xsize (int): Image width, in pixels
            ysize (int): Image height, in pixels
            rescale_factor (int): Rescale down by this factor (useful for PSF compute speed)

        Returns:
            station_locs (np.ndarray): Station locations array,
                                       Array shape is (ysize // rescale_factor, xsize // rescale_factor)
            station_overlay (np.ndarray): Basic image showing detected station locations
                                       Array shape is (ysize, xsize)
        """

        img = 255 - img
        station_overlay = np.zeros_like(img)
        station_locs = np.zeros([ysize // rescale_factor + 1, xsize // rescale_factor + 1])

        min_rad = int(xsize // 40 / 2)
        max_rad = int(xsize // 10 / 2)
        min_dist = 2 * min_rad

        #cv2.HoughCircles(image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)
        #   image: input webcam image size
        #   method: must be cv2.CV_HOUGH_GRADIENT exists
        #   dp: Inverse ratio of the accumulator resolution to the image resolution. this basically affects the min/max radius
        #   minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
        #   circles: set to None
        #   param1: threshold parameter
        #   param2: The smaller it is, the more false circles may be detected.
        #   minRadius: Minimum circle radius
        #   maxRadius: Maximum circle radius
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.1, minDist=min_dist,
                                   param1=50, param2=30, minRadius=min_rad, maxRadius=max_rad)

        if circles is not None:
            for cn,circle in enumerate(circles[0]):
                y,x = circle[1],circle[0]
                try:
                    station_overlay[int(y-5):int(y+5), int(x-5):int(x+5)] = 1
                except:
                    pass
                station_locs[int(y // rescale_factor), int(x // rescale_factor)] = 1
        else:
                station_overlay[int(ysize/2-5):int(ysize/2+5), int(xsize/2-5):int(xsize/2+5)] = 1
                station_locs[int(ysize/2 // rescale_factor), int(xsize/2 // rescale_factor)] = 1
        return station_locs, station_overlay

##############
## Interferometer maths
##############

def gauss_grid(xsize: int, ysize: int) -> np.ndarray:
    """ Create a Gaussian grid to modulate the PSF with

    Args:
        xsize (int): Grid width in pixels
        ysize (int): Grid height in pixels
    """
    # make a 2D Gaussian to modulate the PSF with
    def gauss2d(x0, y0, amp, stdx, stdy):
        return lambda x, y: amp * np.exp(-1. * ((((x - x0)**2.) / (2 * stdx**2.)) +
                                         (((y - y0)**2.) / (2 * stdy**2.))))
    gaussFunc = gauss2d(0., 0., 1., 40., 40.)
    xx = np.arange(xsize) - (xsize / 2)
    yy = np.arange(ysize) - (ysize / 2)
    xv, yv = np.meshgrid(xx, yy)
    gaussGrid = gaussFunc(xv, yv)
    return gaussGrid

def fast_conv(image, psf):
    """ Convolve an image with a PSF (point spread function)

    By the convolution theorem, using '*' to denote convolution operation,
    and '.' to denote point-wise multiplication, we have

        dirty_image = clean_image * psf = InverseFourierTransform ( CLEAN_IMAGE . PSF )

    where CLEAN_IMAGE and PSF are the fourier transforms of clean_image and psf.

    Args:
        image (np.ndarray): Input image.
        psf (np.ndarray): Point spread function

    Returns:
        dirty_image (np.ndarray): 'Dirty' input image after convolution with PSF

    """
    max_size = np.array([np.max([image.shape[0], psf.shape[0]]),
                         np.max([image.shape[1], psf.shape[1]])])
    n = int(2**np.ceil(np.log2(max_size[0])))
    m = int(2**np.ceil(np.log2(max_size[1])))
    imageDirty = np.fft.irfft2(np.fft.rfft2(image, (n, m)) * np.fft.rfft2(psf, (n, m)))
    return imageDirty[psf.shape[0] // 2:image.shape[0] + psf.shape[0] // 2,
                      psf.shape[1] // 2:image.shape[1] + psf.shape[1] // 2]

def compute_psf(station_locs: np.ndarray, xsize: int, ysize: int, gauss_grid: np.ndarray) -> np.ndarray:
    """ Compute the point spread function (PSF) for a given station layout

    The PSF tells us how the telescope would respond to a point source of radiation:
    power from the point will leak, or 'spread' into other pixels.

    The PSF is computed by taking the Fourier transform of the station locations.
    For an interferometer, adding more stations will improve the PSF, making images cleaner.

    Args:
        station_locs (np.ndarray): Station locations, as a 2D numpy array.
        xsize (int): Station image width, in pixels
        ysize (int):  Station image height, in pixels
        gauss_grid (np.ndarray): A Gaussian grid that is modulated with the PSF

    Returns:
        psf_norm (np.ndarray): Point spread function multiplied by Gaussian.
                               Normalized so maximum value is 1.
    """
    psf = np.fft.fftshift(np.abs(np.fft.fft2(station_locs, s=[ysize, xsize]))**2)
    psf_norm = (psf * gauss_grid / psf.max())
    return psf_norm


if __name__ == "__main__":

    CAMERA_DEVICE_INDEX = 0          # For my mac, the webcam is showing up as id=0
    MAXSIZE = 640                    # Maximum image size. If set too high, things get slow.

    # Create a camera object
    cam0 = cv2.VideoCapture(CAMERA_DEVICE_INDEX)
    test_camera(cam0)

    # Set the aspect ratio and compute image shape
    aspect_ratio = get_aspect_ratio(cam0)
    XSIZE = MAXSIZE
    YSIZE = int(MAXSIZE / aspect_ratio)

    # Precompute a Gaussian grid for PSF
    gauss_grid = gauss_grid(XSIZE, YSIZE)

    # Now, start the web app
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # This callback updates the images on a timed interval
    @app.callback(
        Output("plots", "children"),
        Input('timer', 'n_intervals'))
    def update_webcam(t_counter):

        # Capture layout image from camera, and load target sky image
        layout_img = read_from_camera(cam0, XSIZE, YSIZE)
        target_img = load_image('images/gleam.jpg', XSIZE, YSIZE)

        # Detect stations, compute the PSF for the station, and create a dirty (convolved) image
        station_locs, station_overlay = detect_stations(to_greyscale(layout_img), XSIZE, YSIZE)
        psf = compute_psf(station_locs, XSIZE, YSIZE, gauss_grid)
        dirty_img = fast_conv(to_greyscale(target_img), psf)

        # Convert images into Plotly imshows for display in the app
        fig_layout = px.imshow(layout_img, color_continuous_scale='gray', title='Antenna Layout (webcam)')
        fig_target = px.imshow(target_img, color_continuous_scale='gray', title='Actual sky')
        fig_dirty  = px.imshow(dirty_img, color_continuous_scale='gray', title='Observed sky')
        fig_station_locs = px.imshow(station_locs, color_continuous_scale='gray', title='Detected Antennas')

        # Set some plot layout values so it looks nicer
        for fig in (fig_layout, fig_target, fig_dirty, fig_station_locs):
            pdict = {
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'coloraxis_showscale': False,
                'font': {'color': '#ffffff'},
                'margin_l': 10,
                'margin_r': 10,
                'margin_t': 50,
                'margin_b': 10
            }

            fig.update_layout(pdict)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

        # Generate the HTML Table for a 2x2 grid of images
        row1 = html.Tr([html.Td(dcc.Graph(id="antenna_layout", figure=fig_layout)),
                        html.Td(dcc.Graph(id="station_locs", figure=fig_station_locs))])
        row2 = html.Tr([html.Td(dcc.Graph(id="target", figure=fig_target)),
                        html.Td(dcc.Graph(id="dirty", figure=fig_dirty))])
        children = [html.Tbody([row1, row2])]

        return children

    # Set app layout
    # This is the main Dash layout
    app.layout = html.Div([
        html.H4('Interferometer simulator'),
        html.Div(children=update_webcam(0), id='plots'),
        dcc.Interval(
            id='timer',
            interval=1000, # in milliseconds
            n_intervals=0
        )
    ])

    # Finally, start the web app running
    app.run_server(debug=True)