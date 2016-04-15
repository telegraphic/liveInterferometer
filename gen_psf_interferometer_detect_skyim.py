#! /usr/bin/python
"""
Interferometer/PSF simulator

Created by: Jack Hickish
Minor Modifications by: Griffin Foster
Modified 14/06/16 -- Added command entry with curses. Added pause and "persistence" mode

TODO: add color
TODO: adjust detection paramters
TODO: add rotation command
"""

import cv2 #for ubuntu 12.04 install see: http://karytech.blogspot.com/2012/05/opencv-24-on-ubuntu-1204.html
import numpy as np
import time
import sys, optparse
import curses

def cvfast_conv(image,psf):
    max_size = np.array([np.max([image.shape[0],psf.shape[0]]),np.max([image.shape[1],psf.shape[1]])])
    n = cv2.getOptimalDFTSize(max_size[0]*2)
    m = cv2.getOptimalDFTSize(max_size[1]*2)
    imagePad=np.zeros((n,m))
    imagePad[:image.shape[0],:image.shape[1]]=image

    imageDirty=np.fft.irfft2(np.fft.rfft2(imagePad) * np.fft.rfft2(psf, imagePad.shape))
    #print image.shape, psf.shape, n,m
    return imageDirty[psf.shape[0]/2:image.shape[0]+psf.shape[0]/2,psf.shape[1]/2:image.shape[1]+psf.shape[1]/2]
    #return imagePad

def main(stdscr):
    o = optparse.OptionParser()
    o.set_usage('%prog [options]')
    o.set_description(__doc__)
    o.add_option('-i','--input',dest='input', default=None,
        help='Input \'sky\' image, Default: HARDCODED')
    o.add_option('-c','--camera',dest='camera', default=1, type='int',
        help='Camera device ID in /dev/video*, Default: 1')
    o.add_option('-r','--res',dest='res', default=4, type='int',
        help='Resolution factor, increase this value to decrease the resolution, Default: 4')
    opts, args = o.parse_args(sys.argv[1:])
    
    CAMERA_DEVICE_INDEX=opts.camera   #check /dev/, ID is attached to video device (0 is in the internal)
    
    cv2.namedWindow("Antenna Layout", cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow("Target Image", cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow("Point Spread", cv2.CV_WINDOW_AUTOSIZE)
    cv2.namedWindow("Observed Image", cv2.CV_WINDOW_AUTOSIZE)

    cam0 = cv2.VideoCapture(CAMERA_DEVICE_INDEX)
    cam0.read()
    time.sleep(1)
    
    if opts.input is None:
        target_image = cv2.imread('/home/griffin/Downloads/interactiveInterferometer/astro_test_image.jpg')
    else:
        target_image=cv2.imread(opts.input)
    target_img_grey = cv2.cvtColor(target_image,cv2.COLOR_BGR2GRAY)
    target_img_lying = target_img_grey.copy() # show this image to the user.
    # Real computation is done with a saturated image
    target_img_grey[target_img_grey>100] = 255
    target_img_grey[target_img_grey<255] = 0
    
    RESCALE_FACTOR=opts.res #decrease to change the effective resolution
    ysize=480
    xsize=640
    
    #make a 2D Gaussian to modulate the PSF with
    def gauss2d(x0,y0,amp,stdx,stdy):
        return lambda x,y: amp*np.exp(-1.*( (((x-x0)**2.)/(2*stdx**2.)) + (((y-y0)**2.)/(2*stdy**2.)) ))
    gaussFunc=gauss2d(0.,0.,1.,60.,60.)
    xx = np.arange(xsize)-(xsize/2)
    yy = np.arange(ysize)-(ysize/2)
    xv, yv = np.meshgrid(xx, yy)
    gaussGrid=gaussFunc(xv,yv)


    stdscr.nodelay(1) #don't wait for keyboard input when calling getch
    stdscr.clear()
    stdscr.addstr('Controls: Quit (q), Pause (p), Toggle Persistence (r)\n')
    persistence = False
    while(True):
        rv, layout_img = cam0.read()
    
        layout_img_grey = cv2.cvtColor(layout_img, cv2.COLOR_BGR2GRAY)
    
        # set the station locations to zero for each loop, unless we have persistence, in which
        # case leave the ones from the previous iteration in
        if not persistence:
            station_locs = np.zeros([ysize/RESCALE_FACTOR,xsize/RESCALE_FACTOR])
	    station_overlay = np.zeros_like(layout_img_grey)
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
        circles = cv2.HoughCircles(layout_img_grey, cv2.cv.CV_HOUGH_GRADIENT,2,10,None,100,50,5,30)
        if circles is not None:
            for cn,circle in enumerate(circles[0]):
                x,y = circle[1],circle[0]
                try:
                    station_overlay[x-5:x+5,y-5:y+5] = 1
                except:
                    pass
                station_locs[x/RESCALE_FACTOR,y/RESCALE_FACTOR]=1

        # draw white squares at the station locations
        layout_img_grey[station_overlay==1] = 255
        

        psf = np.fft.fftshift(np.abs(np.fft.fft2(station_locs,s=[ysize,xsize]))**2)
        #psf=psf[(ysize/2)-64:(ysize/2)+64,(xsize/2)-64:(xsize/2)+64] #only select the central region of the PSF
        if psf.max() != 0:
            psf *= (gaussGrid/psf.max()) #apply a Gaussian taper to the PSF
    
        #target_arr = target_img_grey[:,:]
        #dirty_arr = cvfast_conv(target_arr,psf)
        dirty_arr = cvfast_conv(target_img_lying, psf)
        
        if dirty_arr.max() != 0:
            dirty_arr /= dirty_arr.max()

        cv2.imshow("Antenna Layout",layout_img_grey)
        cv2.imshow("Target Image", target_img_lying)
        cv2.imshow("Point Spread",psf)
        cv2.imshow("Observed Image",dirty_arr)
    
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
                   time.sleep(0.1)
           elif k == ord('r'):
               if persistence:
                   stdscr.addstr('Turning Persistence off\n')
               else:
                   stdscr.addstr('Turning Persistence on\n')
               persistence = not persistence
               

        cv2.waitKey(1)

cv2.destroyAllWindows()

if __name__ == "__main__":
    curses.wrapper(main)
