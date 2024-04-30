import cv2
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

def test_camera(cam: cv2.VideoCapture):
    rv, img = cam.read()
    if img is None or rv is False:
        raise RuntimeError("Camera not working")

def load_target_image(filepath: str):
    target_image = cv2.imread(filepath)
    target_img_grey = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    target_img_grey[target_img_grey < 100] = 0
    return target_img_grey

def gauss_grid(xsize, ysize):
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
    max_size = np.array([np.max([image.shape[0], psf.shape[0]]),
                         np.max([image.shape[1], psf.shape[1]])])
    n = int(2**np.ceil(np.log2(max_size[0])))
    m = int(2**np.ceil(np.log2(max_size[1])))
    imageDirty = np.fft.irfft2(np.fft.rfft2(image, (n, m)) * np.fft.rfft2(psf, (n, m)))
    return imageDirty[psf.shape[0] // 2:image.shape[0] + psf.shape[0] // 2,
                      psf.shape[1] // 2:image.shape[1] + psf.shape[1] // 2]

def read_from_camera(cam, xsize, ysize):
    for i in range(4):
        rv, layout_img = cam.read()

    layout_img = cv2.resize(layout_img, (ysize, xsize), interpolation=cv2.INTER_CUBIC)
    layout_img_grey = cv2.cvtColor(layout_img, cv2.COLOR_BGR2GRAY)
    layout_img_grey = np.fliplr(layout_img_grey)
    return layout_img_grey

def detect_stations(img, ysize, xsize, rescale_factor=4):
        #cv2.HoughCircles(image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)
        #   image: input webcam image size
        #   method: only cv.CV_HOUGH_GRADIENT exists
        #   dp: Inverse ratio of the accumulator resolution to the image resolution. this basically affects the min/max radius
        #   minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
        #   circles: set to None
        #   param1: threshold parameter
        #   param2: The smaller it is, the more false circles may be detected.
        #   minRadius: Minimum circle radius
        #   maxRadius: Maximum circle radius
        img = 255 - img
        station_overlay = np.zeros_like(img)
        station_locs = np.zeros([ysize // rescale_factor, xsize // rescale_factor])

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 20, param1=50, param2=30, minRadius=50, maxRadius=128)

        if circles is not None:
            print(len(circles[0]))
            for cn,circle in enumerate(circles[0]):
                x,y = circle[1],circle[0]

                try:
                    station_overlay[int(x-5):int(x+5),int(y-5):int(y+5)] = 1
                except:
                    pass
                station_locs[int(x // rescale_factor), int(y // rescale_factor)]=1

        return station_locs, station_overlay

def compute_psf(station_locs, xsize, ysize, gauss_grid):
        psf = np.fft.fftshift(np.abs(np.fft.fft2(station_locs, s=[ysize, xsize]))**2)
        psf_norm = (psf * gauss_grid / psf.max())
        return psf_norm



CAMERA_DEVICE_INDEX = 0
XSIZE, YSIZE = (480, 640)

cam0 = cv2.VideoCapture(CAMERA_DEVICE_INDEX)

test_camera(cam0)
gauss_grid = gauss_grid(XSIZE, YSIZE)

target_img = load_target_image('astro_test_image.jpg')
layout_img = read_from_camera(cam0, XSIZE, YSIZE)
station_locs, station_overlay = detect_stations(layout_img, XSIZE, YSIZE)
psf = compute_psf(station_locs, XSIZE, YSIZE, gauss_grid)
dirty_img = fast_conv(target_img, psf)

fig_layout = px.imshow(layout_img, binary_string=True)
fig_target = px.imshow(target_img, binary_string=True)
fig_dirty  = px.imshow(dirty_img, binary_string=True)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

@app.callback(
    Output("plots", "children"),
    Input('timer', 'n_intervals'))
def update_webcam(t_counter):
    layout_img = read_from_camera(cam0, XSIZE, YSIZE)
    target_img = load_target_image('astro_test_image.jpg')
    layout_img = read_from_camera(cam0, XSIZE, YSIZE)
    station_locs, station_overlay = detect_stations(layout_img, XSIZE, YSIZE)
    psf = compute_psf(station_locs, XSIZE, YSIZE, gauss_grid)
    dirty_img = fast_conv(target_img, psf)

    fig_layout = px.imshow(layout_img, color_continuous_scale='gray')
    fig_target = px.imshow(target_img, color_continuous_scale='gray')
    fig_dirty  = px.imshow(dirty_img, color_continuous_scale='gray')
    fig_station_locs = px.imshow(station_locs, color_continuous_scale='gray')

    for fig in (fig_layout, fig_target, fig_dirty, fig_station_locs):
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    row1 = html.Tr([html.Td(dcc.Graph(id="antenna_layout", figure=fig_layout)),
                    html.Td(dcc.Graph(id="station_locs", figure=fig_station_locs))])
    row2 = html.Tr([html.Td(dcc.Graph(id="target", figure=fig_target)),
                    html.Td(dcc.Graph(id="dirty", figure=fig_dirty))])

    children = [html.Tbody([row1, row2])]

    return children

app.layout = html.Div([
    html.H4('Interferometer simulator'),
    html.Div(children=update_webcam(0), id='plots'),
    dcc.Interval(
        id='timer',
        interval=1000, # in milliseconds
        n_intervals=0
    )
])


app.run_server(debug=True)
