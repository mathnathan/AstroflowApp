import numpy as np

# I was tired of hardcoding different sized kernels when I wanted to try
# different things so I just wrote this routine to create them on the fly
def createSobelKernel(size, axis=0):

    print "-- Creating Sobel Kernel of size %d..." % (size)

    assert size % 2 == 1, "\nKernel size must be odd!"

    magic = size / 2
    kernel = list()
    for i in xrange(-magic, magic + 1):
        row = list()
        if i < 0:
            pivot = i - magic
            for j in xrange(-magic, magic + 1):
                e = pivot + np.abs(j)
                row.append(e)
            kernel.append(row)
        elif i == 0:
            row = [0] * size
            kernel.append(row)
        else:
            pivot = i + magic
            for j in xrange(-magic, magic + 1):
                e = pivot - np.abs(j)
                row.append(e)
            kernel.append(row)

    if axis == 0:
        return np.asarray(kernel)
    elif axis == 1:
        return np.asarray(kernel).T

# Standard convolution of derivative kernel with data
def firstYderiv(img, kernelSize=3):

    print "-- Calculating Derivative"

    s = kernelSize
    assert s % 2 == 1, "Kernel size must be odd!"
    ky = createSobelKernel(s, axis=0)

    h, w = img.shape
    yDeriv = np.zeros(img.shape)
    r = s / 2 # radius
    imgCpy = img.copy()
    # I do not do any padding for these. No derivative is calculated for any
    # pixel that is r pixels in from the border. This should not be an issue for
    # this application.
    for j in xrange(r, h - r):
        for i in xrange(r, w - r):
            # Here we just extract the submat from the data equal to the size of
            # the derivative kernel
            submat = imgCpy[j - r: j + r + 1, i - r: i + r + 1]
            # Then we 'convolve' the submatrix and the kernel
            val = (ky * submat).sum()
            # Lastly we assign that value to the current pixel
            yDeriv[j, i] = val

    return yDeriv


# The x derivative is the same as the y. We just calculate it on the image's
# transpose
def firstXderiv(img, kernelSize=3):

    xDeriv = firstYderiv(img.T, kernelSize)

    return xDeriv.T

def makeGauss(center=[0.0,0.0], std=[1.0,1.0], amp=1.0, noiseAmp=None):
    """
    Generates a Gaussian function centered at (center[0], center[1]), with x and
    y standard deviation of (std[0], std[1]), an amplitude of 'amp', and noise
    amplitude 'noiseAmp'. This is the returned to be evaluated at any ordered
    pair of real numbers.

    Parameters
    ----------
    center : array_like, optional
        An iterable containing the x and y locations for the Gaussian function
        to be centered about.
    std : array_like, optional
        An iterable containing the x and y standard deviations for the exponent
        of the Gaussian function.
    amp : float, optional
        Specify the amplitude of the gaussian at its center. The default is 1.
    noiseAmp : float, optional
        The amplitude of the noise to be added to the function's evaluations.
        The default is 0.

    Returns
    -------
    gauss : function
        A gaussian function with the properties specified by the parameters above.
        It can be evaluated at any (x,y) location.
    """

    x0 = center[0]; y0 = center[1]
    sx = std[0]*np.sqrt(2); sy = std[1]*np.sqrt(2)
    a = amp; nA = noiseAmp

    from np.random import random as r
    if noiseAmp is None:
        return lambda x,y: a*np.exp(-(((x-x0)/sx)**2+((y-y0)/sy)**2))
    else:
        return lambda x,y: a*np.exp(-(((x-x0)/sx)**2+((y-y0)/sy)**2))+r()*nA

def fitSurface(Xpts, Ypts, Zpts):

    from scipy.interpolate import RectBivariateSpline

    print "-- Fitting Surface..."

    # Downsample for computational efficiency
    factor = 2
    downZpts = Zpts[::factor,::factor]
    downYpts = Ypts[::factor,::factor]
    downXpts = Xpts[::factor,::factor]

    # Now fit a 2D cubic spline through the data for continuity
    xpts = downXpts[:,0]; xmin = xpts.min(); xmax = xpts.max()
    ypts = downYpts[0]; ymin = ypts.min(); ymax = ypts.max()
    f = RectBivariateSpline(xpts, ypts, downZpts, s=0)

    return f

def dctFilter(data, cutOff=None):

    #print "-- Filtering the Data..."

    dim = data.ndim
    assert dim==2 or dim==3, "data must be 2 or 3 dimensional"

    if dim == 3:
        shape = data.shape
        filteredData = np.ndarray(shape)
        for indx, frame in enumerate(data):

            # Transform everything into the frequency domain
            dctData = dct(dct(frame.T).T)

            # Determine the cutoff point
            if cutOff is None:
                cutOff = np.max(shape)*0.2 # Arbitrarily chosen to be 20%

            # Removing noise is done by setting the higher modes to zero
            zdim, ydim, xdim = shape
            ylocs, xlocs = np.mgrid[0:ydim:1, 0:xdim:1]
            locs = np.where(xlocs*xlocs+ylocs*ylocs > cutOff*cutOff)
            dctData[locs] = 0.0

            # Trasnform back into Cartesian space
            cartData = dct(dct(dctData.T, 3).T, 3)

            # Translate back to zero
            #cartData -= cartData.min()
            # Scale the signal back to original scale
            #filteredData[indx] = cartData/(4*xdim*ydim)
            filteredData[indx] = cartData

    else:

        # Transform everything into the frequency domain
        dctData = dct(dct(data.T).T)

        # Determine the cutoff point
        if cutOff is None:
            cutOff = np.max(data.shape)*0.2 # Arbitrarily chosen to be 20%

        # Removing noise is done by setting the higher modes to zero
        ydim, xdim = data.shape
        ylocs, xlocs = np.mgrid[0:ydim:1, 0:xdim:1]
        locs = np.where(xlocs*xlocs+ylocs*ylocs > cutOff*cutOff)
        dctData[locs] = 0.0

        # Trasnform back into Cartesian space
        filteredData = dct(dct(dctData.T, 3).T, 3)

        # Translate back to zero
        filteredData -= filteredData.min()
        # Scale the signal back to original scale
        filteredData /= 4*xdim*ydim

    return filteredData

def gaussianFilter(images, frames=[0,None], disk=(5,5)):
# filter all images with a Gaussian filter 5x5
# filter images in frames[0] to frames[1] (included)
# change images in place

    #print "Gaussian Filter", disk
    output = np.ndarray(np.shape(images))
    if frames[1] == None:
        frames[1] = len(images)

    for i,frame in enumerate(images[frames[0]:frames[1]]):
        # not sure what sigmaX,sigmaY does. It is the std of the Gaussian
        output[i,:,:] = cv2.GaussianBlur(frame, ksize=disk, sigmaX=0, sigmaY=0)

    return output

def roiTimeSeries(data, boxes, times):

    timeSeriesList = []
    rois = zip(times,boxes)
    for roi in rois:
        beg = roi[0][0]; end = roi[0][1]
        x0 = roi[1][0]; y0 = roi[1][1]
        x1 = roi[1][2]; y1 = roi[1][3]
        series = np.ndarray((end-beg))
        for time,frame in enumerate(data[beg:end]):
            pixels = frame[y0:y1, x0:x1]
            series[time] = np.average(pixels)
        timeSeriesList.append(series)

    return timeSeriesList


class Chebyshev2d:

    from numpy.polynomial import chebyshev as cheb

    def __init__(self, x, y, z, degs):

        self.degrees = degs
        self.xpts = x
        self.ypts = y
        self.zpts = z
        self.xlims = [min(x),max(x)]
        self.ylims = [min(y),max(y)]
        self.zlims = [min(z),max(z)]
        self.__buildCheby()


    def __buildCheby(self):

        vander2d = cheb.chebvander2d(self.xpts, self.ypts, self.degrees)
        coefs, residue, rank, s = lstsq(vander2d, self.zpts)
        self.coefs = coefs.reshape(self.degrees[0]+1, self.degrees[1]+1)


    def __call__(self, x):

            return cheb.chebval2d(x[0], x[1], self.coefs)


    def grid(self, x):

            return cheb.chebgrid2d(x[0], x[1], self.coefs)


class Legendre2d:

    from numpy.polynomial import legendre as legend

    def __init__(self, x, y, z, degs):

        self.degrees = degs
        self.xpts = x
        self.ypts = y
        self.zpts = z
        self.xlims = [min(x),max(x)]
        self.ylims = [min(y),max(y)]
        self.zlims = [min(z),max(z)]
        self.__buildLegend()


    def __buildLegend(self):

        vander2d = legend.legvander2d(self.xpts, self.ypts, self.degrees)
        coefs, residue, rank, s = lstsq(vander2d, self.zpts)
        self.coefs = coefs.reshape(self.degrees[0]+1, self.degrees[1]+1)


    def __call__(self, x):

            return legend.legval2d(x[0], x[1], self.coefs)


    def grid(self, x):

            return legend.leggrid2d(x[0], x[1], self.coefs)
