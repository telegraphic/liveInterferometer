#! /usr/bin/python
"""
Interferometer/PSF simulator

Created by: Jack Hickish
Minor Modifications by: Griffin Foster
Modified 14/06/16 -- Added command entry with curses. Added pause and "persistence" mode
Modified Jan 16th 2019 -- CJ, Edgar, + Adam edit. Plot 3 images in one window. 
                          Could not get last one to show in same window
                          Attempts saved in "camera_window_attempt"  

TODO: add color
TODO: adjust detection paramters
TODO: add rotation command
"""

# For ubuntu 12.04 install see: http://karytech.blogspot.com/2012/05/opencv-24-on-ubuntu-1204.html
import cv2 
import numpy as np
import time
import sys
import optparse
import curses
import os


def fast_conv(image, psf):
    max_size = np.array([np.max([image.shape[0], psf.shape[0]]),
                         np.max([image.shape[1], psf.shape[1]])])
    n = int(2**np.ceil(np.log2(max_size[0])))
    m = int(2**np.ceil(np.log2(max_size[1])))
    imageDirty = np.fft.irfft2(np.fft.rfft2(image, (n, m)) * np.fft.rfft2(psf, (n, m)))
    return imageDirty[psf.shape[0] // 2:image.shape[0] + psf.shape[0] // 2,
                      psf.shape[1] // 2:image.shape[1] + psf.shape[1] // 2]


def main(stdscr, opts, args):

    # check /dev/, ID is attached to video device (0 is in the internal)
    CAMERA_DEVICE_INDEX = opts.camera

    #cv2.namedWindow("Antenna Layout", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("Target Image", cv2.WINDOW_AUTOSIZE)
    
    #cv2.namedWindow("testhoriz1", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("testhoriz2", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("Point Spread", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Observed Image", cv2.WINDOW_NORMAL)

    cam0 = cv2.VideoCapture(CAMERA_DEVICE_INDEX)

    if opts.input is None:
        infile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'astro_test_image.jpg')
        target_image = cv2.imread(infile)
    else:
        target_image = cv2.imread(opts.input)
    target_img_grey = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    target_img_grey[target_img_grey < 100] = 0
    

    RESCALE_FACTOR = opts.res
    ysize = opts.xsize
    xsize = opts.ysize

    # make a 2D Gaussian to modulate the PSF with
    def gauss2d(x0, y0, amp, stdx, stdy):
        return lambda x, y: amp * np.exp(-1. * ((((x - x0)**2.) / (2 * stdx**2.)) +
                                         (((y - y0)**2.) / (2 * stdy**2.))))
    gaussFunc = gauss2d(0., 0., 1., 40., 40.)
    xx = np.arange(xsize) - (xsize / 2)
    yy = np.arange(ysize) - (ysize / 2)
    xv, yv = np.meshgrid(xx, yy)
    gaussGrid = gaussFunc(xv, yv)

    stdscr.nodelay(1)  # don't wait for keyboard input when calling getch
    stdscr.clear()
    stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')
    persistence = False
    window = False
    while(True):
        # Grab 4 images. The images are buffered, so this helps
        # get one that is relatively recent. My cv2 doesn't
        # seem to support reducing the size of the buffer.
        for i in range(4):
            rv, layout_img = cam0.read()

        layout_img = cv2.resize(layout_img, (ysize, xsize), interpolation=cv2.INTER_CUBIC)
        layout_img_grey = cv2.cvtColor(layout_img, cv2.COLOR_BGR2GRAY).T
        layout_img_grey = np.fliplr(layout_img_grey)
        # set the station locations to zero for each loop, unless we have persistence, in which
        # case leave the ones from the previous iteration in
        if not persistence:
            station_overlay = np.zeros_like(layout_img_grey)
        station_locs = np.zeros([ysize // RESCALE_FACTOR, xsize // RESCALE_FACTOR])

        method = cv2.HOUGH_GRADIENT  # only method that exists
        dp = 2  # Inverse ratio of the accumulator resolution to the image resolution. this basically affects the min/max radius
        minDist = 50  # Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
        circles = None  # set to None
        param1 = 50  # threshold parameter
        param2 = 30  # The smaller it is, the more false circles may be detected.
        minRadius = 15  # Minimum circle radius
        maxRadius = 30  # Maximum circle radius
        circles = cv2.HoughCircles(layout_img_grey, method, dp, minDist, circles,
                                   param1, param2, minRadius, maxRadius)
        if circles is not None:
            for cn, circle in enumerate(circles[0]):
                x, y = int(circle[1]), int(circle[0])
                try:
                    station_overlay[x - 5:x + 5, y - 5:y + 5] = 1
                except:
                    pass
                station_locs[x // RESCALE_FACTOR, y // RESCALE_FACTOR] = 1

        # draw white squares at the station locations
        layout_img_grey[station_overlay == 1] = 255

        if not persistence:
            psf = np.zeros([ysize, xsize])

        psf += np.fft.fftshift(np.abs(np.fft.fft2(station_locs, s=[ysize, xsize]))**2)

        if psf.max() != 0:
            if window:
                bls = np.fft.ifft2(np.fft.fftshift(psf))
                abs_bls = np.abs(bls)
                abs_bls[0, 0] = 0
                abs_bls[abs_bls < 0.5] = 1e9
                win_psf = np.abs(np.fft.fftshift(np.fft.fft2(bls / abs_bls,
                                                             s=[ysize, xsize])))
                psf_norm = (win_psf * gaussGrid / win_psf.max())
           # apply Gaussian taper to the PSF
            else:
                psf_norm = (psf * gaussGrid / psf.max())  
        else:
            psf_norm = psf

        dirty_arr = fast_conv(target_img_grey, psf_norm)

        if dirty_arr.max() != 0:
            dirty_arr /= dirty_arr.max()


        # stacking images into one window instead of 4. one camera will not work; see note at top
        numpy_horiz_concat1= np.concatenate((target_img_grey, dirty_arr), axis=1)
        numpy_horiz_concat2 = np.concatenate((layout_img_grey, psf_norm), axis=1)		
	
        cat1rs = cv2.resize(numpy_horiz_concat1, (1200, 500))
        cat2rs = cv2.resize(numpy_horiz_concat2, (1200, 500))

        numpy_vert_concat= np.concatenate((cat1rs, cat2rs), axis=0)
        layout_img_grey = cv2.resize(layout_img_grey, (600, 500))			
        
        # showing windows
        cv2.imshow("Target, Observed, PSF", numpy_vert_concat)
        cv2.imshow("Antenna Layout", layout_img_grey)
        #cv2.imshow("Target Image", target_img_grey)
        #cv2.imshow("horiz concat", numpy_horiz_concat2)
        #cv2.imshow("Point Spread", psf_norm)
        #cv2.imshow("Observed Image", dirty_arr)

        # Key command handling
        k = stdscr.getch()
        if k != -1:
            stdscr.clear()
            stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')
            if k == ord('q'):
                break
            elif k == ord('p'):
                stdscr.addstr('Paused. Press any key to continue...\n')
                while(True):
                    k2 = stdscr.getch()
                    if k2 != -1:
                        stdscr.addstr('Unpausing...\n')
                        break
                    cv2.waitKey(1)
                    time.sleep(0.25)
            elif k == ord('r'):
                if persistence:
                    stdscr.addstr('Turning Persistence off\n')
                else:
                    stdscr.addstr('Turning Persistence on\n')
                persistence = not persistence
            elif k == ord('w'):
                if window:
                    stdscr.addstr('Turning window off\n')
                else:
                    stdscr.addstr('Turning window on\n')
                window = not window

        cv2.waitKey(1)

cv2.destroyAllWindows()

if __name__ == "__main__":
    o = optparse.OptionParser()
    o.set_usage('%prog [options]')
    o.set_description(__doc__)
    o.add_option('-i', '--input', dest='input', default=None,
                 help='Input \'sky\' image, Default: astro_test_image.jpg')
    o.add_option('-c', '--camera', dest='camera', default=1, type='int',
                 help='Camera device ID in /dev/video*, Default: 1')
    o.add_option('-r', '--res', dest='res', default=4, type='int',
                 help='Resolution factor, increase this value to decrease the'
                      'resolution, Default: 4')
    o.add_option('-x', '--xsize', dest='xsize', default=640, type='int',
                 help='Horizontal dimension of images')
    o.add_option('-y', '--ysize', dest='ysize', default=480, type='int',
                 help='Vertical dimension of images')
    opts, args = o.parse_args(sys.argv[1:])
    curses.wrapper(main, opts, args)

