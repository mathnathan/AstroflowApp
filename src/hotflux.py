import numpy as np
import cv2
from scipy import interpolate
import readWrite as rw
import sys, os
from tqdm import tqdm  # time loops


class HotFlux():


    def __init__(self, inFileName):

        self.dataPath = inFileName.rstrip('/')
        self.basePath, self.fullFilename = os.path.split(self.dataPath)
        self.filename, self.ext = os.path.splitext(self.fullFilename)
        self.filenameXflow = os.path.join(self.basePath, self.filename) + "_flowx.npy"
        self.filenameYflow = os.path.join(self.basePath, self.filename) + "_flowy.npy"
        self.data = rw.loadData(self.dataPath)
        if self.data.ndim == 4:  # I might want to undo this later....
            # If this is true, this usually means the 4th dimension is multiple
            # experiments. For now I just average all experiments together
            self.data = np.average(self.data, 0)  # Average the experiments
        self.avgData = np.average(self.data, 0)
        self.shape = self.data.shape
        self.createFlow()


    def createFlow(self):
        """Create the flow fileds using Farneback's optical flow algorithm from
        the OpenCV library"""

        print "-- Calculating Flow..."
        if os.path.exists(self.filenameXflow) and os.path.exists(self.filenameYflow):
            print "\t- Flow Already Calculated. Loading from memory...\n"
            self.xflow = np.load(self.filenameXflow)
            self.yflow = np.load(self.filenameYflow)
        else:  # Now calculate flow
            zdim, ydim, xdim = self.data.shape
            #blurData = gauss(self.data, (0.9,0.65,0.65))
            blurData = self.data  # No blur for now...

            self.xflow = np.zeros((zdim-1, ydim, xdim)) # Store the xflow vectors in here
            self.yflow = np.zeros((zdim-1, ydim, xdim)) # Store the yflow vectors in here
            prev = blurData[0]
            pyr_scale = 0.5
            levels = 5
            winSz = 16
            itrs = 5
            polyN = 4
            polyS = 0.8
            flg = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            optFlow = cv2.calcOpticalFlowFarneback
            calcFlow = None
            if int(cv2.__version__[0]) == 3:
                calcFlow = lambda p, c: optFlow(p, c, None, pyr_scale, levels, winSz, itrs, polyN, polyS, flg)
            else:
                calcFlow = lambda p, c: optFlow(p, c, pyr_scale, levels, winSz, itrs, polyN, polyS, flg)
            for i,curr in enumerate(blurData[1:]):
                flow = calcFlow(prev, curr)
                self.xflow[i] = flow[:,:,0]
                self.yflow[i] = flow[:,:,1]
                prev = curr
            np.save(self.filenameXflow, self.xflow)
            np.save(self.filenameYflow, self.yflow)

        return (self.xflow, self.yflow)


    def findHotspots(self, inKnots, xtipsIn=None, ytipsIn=None, numPpts=200, beg=0, end=-1):

        if xtipsIn is None:
            xtips = self.xflow[beg:end]
        else:
            xtips = xtipsIn.copy()
        if ytipsIn is None:
            ytips = self.yflow[beg:end]
        else:
            ytips = ytipsIn.copy()

        dzim, ydim, xdim = xtips.shape
        knots = self.formatKnots(inKnots)
        tck, u = interpolate.splprep([knots[:,0], knots[:,1]], s=0.0)
        samplePts = np.linspace(0,1,numPpts)
        pts = np.array(interpolate.splev(samplePts, tck, der=0))
        derivs = np.array(interpolate.splev(samplePts, tck, der=1))
        derivs /= np.sqrt((derivs*derivs).sum(axis=0))
        numFrames = xtips.shape[0]
        fluxVtime = np.ndarray((numFrames, numPpts))
        results = []
        flowVecs = zip(xtips, ytips)
        hotspots = []
        for i,((x,y),(dx,dy)) in enumerate(zip(pts.T,derivs.T)):
            results.append({'x': x, 'y': y, 'dx': dx, 'dy': dy,
                            'xflow': [], 'yflow': [], 'flowVtime': [],
                            'hotspots': []})
            for j,(xtip, ytip) in enumerate(flowVecs):  # For each frame

                # First find the vector from the flow field corresponding to this point
                # Find the bounding x indices for the vector tip
                #xindx = np.argmin(np.absolute(xRange-x))
                nearestXVal = np.round(x)  #closest euclidean point
                if nearestXVal > x or nearestXVal == xdim-1:
                    x0 = int(nearestXVal - 1)
                    x1 = int(nearestXVal)
                else:
                    x0 = int(nearestXVal)
                    x1 = int(nearestXVal + 1)
                #print "X values at (%f,%f) = (%f,%f)" % (x0indx,x1indx,x0,x1)
                # Find the bounding y indices for the vector
                #yindx = np.argmin(np.absolute(yRange-y))
                nearestYVal = np.round(y)  # closest euclidean point
                if nearestYVal > y or nearestYVal == ydim-1:
                    y0 = int(nearestYVal - 1)
                    y1 = int(nearestYVal)
                else:
                    y0 = int(nearestYVal)
                    y1 = int(nearestYVal + 1)

                # Now perform rectilinear interpolation to find best vector to
                # take the inner product with
                xpts = (x0,x1)
                ypts = (y0,y1)

                if 1:
                    # AVERAGING THE VECTORS BEFORE BILINEAR INTERPOLATION TO
                    # IMPROVE RESULTS (Hopefully)
                    zptsX0 = xtip[y0-1:y1+1,x0-1:x1+1].mean()
                    zptsX1 = xtip[y0-1:y1+1,x0:x1+2].mean()
                    zptsX2 = xtip[y0:y1+2,x0-1:x1+1].mean()
                    zptsX3 = xtip[y0:y1+2,x0:x1+2].mean()
                    zptsY0 = ytip[y0-1:y1+1,x0-1:x1+1].mean()
                    zptsY1 = ytip[y0-1:y1+1,x0:x1+2].mean()
                    zptsY2 = ytip[y0:y1+2,x0-1:x1+1].mean()
                    zptsY3 = ytip[y0:y1+2,x0:x1+2].mean()
                    zptsX = np.array(((zptsX0, zptsX1),(zptsX2, zptsX3)))
                    zptsY = np.array(((zptsY0, zptsY1),(zptsY2, zptsY3)))
                    # DONE AVERAGING
                else:
                    zptsX = xtip[y0:y1+1,x0:x1+1]
                    zptsY = ytip[y0:y1+1,x0:x1+1]

                xvec = self.bilinearInterp(xpts, ypts, zptsX, x, y)
                yvec = self.bilinearInterp(xpts, ypts, zptsY, x, y)
                results[i]['xflow'].append(xvec)
                results[i]['yflow'].append(yvec)
                slope = np.array((dx, dy)) # This is also the derivative
                flow = np.array((xvec, yvec))
                flowAlong = np.dot(flow, slope)
                results[i]['flowVtime'].append(flowAlong)
                fluxVtime[j,i] = flowAlong

        for i,flx in enumerate(fluxVtime):
            #print "Plot #%d" % (i)
            zero_crossings = np.where(np.diff(np.signbit(flx)))[0]
            #print "Zero Crossing Indices"
            #print "\t", zero_crossings
            #print "Physical Coordinates"
            for zero in zero_crossings:
                results[zero]['hotspots'].append(i)
                #print "\t%f -> (%f,%f)" % (zero, pts[0,zero], pts[1,zero])

        return results


    def calcCentroid(self, array=None):

        if array is None:
            try:
                return self.centroid
            except:
                avg = self.avgData.copy()
        else:
            avg = array.copy()

        h, w = avg.shape
        # Threshold the data to average only the high intensity values
        avg[avg < np.percentile(avg, 99.9)] = 0.0

        xvec = np.arange(1,w+1,1)
        xbars = np.ndarray(h)
        for i in range(h):
            arrSlice = np.array(avg[i,:], dtype=np.float)
            asum = arrSlice.sum()
            if asum == 0:
                xbars[i] = 0
            else:
                xbars[i] = np.dot(arrSlice,xvec)/asum
        xbar = (xbars.sum()/len(xbars.nonzero()[0]))-1

        yvec = np.arange(1,h+1,1)
        ybars = np.ndarray(w)
        for j in range(w):
            arrSlice = np.array(avg[:,j], dtype=np.float)
            asum = arrSlice.sum()
            if asum == 0:
                ybars[j] = 0
            else:
                ybars[j] = np.dot(arrSlice,yvec)/asum
        ybar = (ybars.sum()/len(ybars.nonzero()[0]))-1

        self.centroid = np.array((xbar, ybar))

        return self.centroid


    def bilinearInterp(self, xpts, ypts, zpts, x, y):

        x0, x1 = xpts
        y0, y1 = ypts
        z0, z1, z2, z3 = zpts.flatten()

        tx = (x1 - x)/float(x1 - x0)
        ty = (y1 - y)/float(y1 - y0)

        return (z0-z1-z2+z3)*tx*ty+(z2-z3)*tx+(z1-z3)*ty+z3

        #fx0 = zpts[0,0]*tx + zpts[0,1]*(1-tx)
        #fx1 = zpts[1,0]*tx + zpts[1,1]*(1-tx)
        #fx = fx0*ty + fx1*(1-ty)
        #return fx


    def verifyData(self, knots, xtail, ytail):

            rows, cols = knots.shape
            if rows == 2:
                xmin = np.min(knots[0]); xmax = np.min(knots[0])
                ymin = np.min(knots[1]); ymax = np.min(knots[1])
            elif cols == 2:
                xmin = np.min(knots[:,0]); xmax = np.min(knots[:,0])
                ymin = np.min(knots[:,1]); ymax = np.min(knots[:,1])
            else:
                assert False, "Incoming interpolant knots must be 2 dimensional"

            if xmin < np.min(xtail[0]) or xmax > np.max(xtail[0]):
                return False
            if ymin < np.min(ytail[:,0]) or ymax > np.max(ytail[:,0]):
                return False

            return True

    def formatKnots(self, data):
        """
        Here we turn everything into a numpy array. Also, regardless of the shape
        of the array containing the points, we convert it to have shape = (n,2).
        In this way, data[i] returns the ith point allowing for easy pointwise
        vector arithmetic.
        """

        tp = type(data)
        if tp == tuple or tp == list:
            data = np.array(data)
        elif tp != np.ndarray:
            assert False, "Incoming points must be a tuple, list, or ndarray"

        rows, cols = data.shape
        if rows == 2:
            return data.T
        elif cols == 2:
            return data
        else:
            assert False, "Incoming arrays must be 2 dimensional"

    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    def calcFluxLinearTEST(self, inKnots, calDataIn=None, xtailIn=None, ytailIn=None, xtipsIn=None, ytipsIn=None, width=1, beg=0, end=-1):
        print "enter calcFluxLinear"
        #print type(inKnots)
        #print len(inKnots); 
        #print np.shape(inKnots); quit()
        """
        Calculate the flux of a path specified by knot points over a series of
        vector vields determined by xtips and ytips. For subpixel accuracy, the
        path will be linearly interpolated and each vector field will be
        bilinearly interpolated.

        Parameters
        ----------
        inKnots : array-like
            This is the sequence of knot points used to create the piecewise linear interpolant.
            It must be either a tuple, a list, or an ndarray and must be two dimensional of shape
            (2,n) or (n,2) where each point is specified by (x,y).
        calDataIn : array-like
            The actual calcium intensity data. Should have shape (t,m,n) where t is the number of
            time steps and m, n are the rows and columns respectively. The shape should match all
            of the vector information as well.
        xtailIn : ndarray
            A 2 dimensional mesh specifying the x coordinates of the grid for each vector field.
            This must match the shape of ytail and each xtip, ytip. In addition, the x coordinates
            of the knots must be contained within the domain of xtail.
        ytailIn : ndarray
            A 2 dimensional mesh specifying the y coordinates of the grid for each vector field.
            This must match the shape of xtail and each xtip, ytip. In addition, the y coordinates
            of the knots must be contained within the domain of ytail.
        xtipsIn : ndarray
            Each element represents the x component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each ytip
        ytipsIn : ndarray
            Each element represents the y component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each xtip
        width : scalar
            Approximate thickness of process in pixels. Assumed to be 10
        beg : scalar
            Beginning index to slice the time series for analysis
        end : scalar
            Ending index to slice the time series for analysis

        Returns
        -------
        flux : ndarray
            A time series representing the associated flux along the path specified by
            inKnots over the vector fields specified by xtips and ytips.
        """

        if calDataIn is None:
            calData = self.data[beg:end]
        else:
            calData = calDataIn.copy()
        if xtipsIn is None:
            xtips = self.xflow[beg:end]
        else:
            xtips = xtipsIn.copy()
        if ytipsIn is None:
            ytips = self.yflow[beg:end]
        else:
            ytips = ytipsIn.copy()
        if xtailIn is None:
            zdim, ydim, xdim = self.shape
            trash, xtail = np.mgrid[0:ydim, 0:xdim]
        else:
            xtail = xtailIn.copy()
        if ytailIn is None:
            zdim, ydim, xdim = self.shape
            ytail, trash = np.mgrid[0:ydim, 0:xdim]
        else:
            ytail = ytailIn.copy()

        avgImg = self.avgData
        #avgImg[avgImg < np.percentile(avgImg, 60)] = 0
        center = self.calcCentroid(avgImg) # The centroid of the calcium image!
        knots = self.formatKnots(inKnots)
        #print np.shape(knots); quit()
        width = float(width) 
        #print("width= ", width); quit() # 1

        # leave knots as is. 
        #knots = knots[::len(knots)/50]  # Downsample the number of knots

        numKnots = len(knots)

        # knots have special significants. 
        # Use an array of points, rather than knots
        numPts = numKnots
        pts    = knots   # (186, 2)

        # We want the slope to always be pointing toward the soma. This forms
        # the sign convention that positive flux is always inward or towards the
        # soma, and negative flux is always outward or away from the soma.
        # We assume the path was drawn inward (clicked far from soma and dragged
        # toward it). If this is the case then the slope is already toward the soma
        # when calculating along the parameterization direction. Below we test if
        # it was drawn outward. If it was drawn outward we just reverse the order
        # of the knots so that it was drawn inward! Easy fix
        if np.linalg.norm(center-knots[-1]) > np.linalg.norm(center-knots[0]): # Drawn outward
            knots = knots[::-1] # So we reverse the order. Now it is drawn inward

        assert self.verifyData(knots, xtail, ytail), "Interpolation knots must be contained within the vector field!"
        xRange = xtail[0] # Used for finding bounding indices around each point (bilinear interp)
        yRange = ytail[:,0] # Used for finding bounding indices around each point (bilinear interp)
        ydim, xdim = xtail.shape # Used for finding bounding indices around each point (bilinear interp)
        numSteps_dt = 6.0 # Number of points for stepping between the knots points
        numSteps_ds = 5.0 # Number of points for stepping along the line perpendicular to path
        fluxPts = np.zeros(len(xtips))
        xvecs = []
        yvecs = []

        # list of points

        d = width/2.0 # integral for flux goes from -d to d
        ds = width/numSteps_ds # The stepsize for above integral
        dt = 1.0/numSteps_dt # stepsize for integral along path

        for i,(xtip,ytip) in tqdm(enumerate(zip(xtips,ytips))):  # For each frame
            pt0 = knots[0]
            totFlux = 0
            for j,pt1 in tqdm(enumerate(knots[1:])):  # For each knot point
                # Linearly interpolating between knot points. Therefore the gradient
                # and slope will not change for the partitioning points along the
                # way between knots
                slope = pt1-pt0 # This is also the derivative
                slopeNorm = np.linalg.norm(slope)
                grad = np.array((-slope[1], slope[0]))/slopeNorm
                for k,t in tqdm(enumerate(np.linspace(0,1,numSteps_dt))):  # For each subknot
                    pt = pt0 + slope*t # Step along the line from pt0 to pt1
                    flux = 0
                    spts = np.arange(-d+ds/2.0,d,ds) # Quadrature points for the midpoint rule:
                    for l,s in tqdm(enumerate(spts)): # For each point along perpendicular line
                        x, y = pt + grad*s
                        # At this (x,y), find the bounding indices in xtail and ytail
                        # First find the bounding x indices for the flow vector
                        xindx = np.argmin(np.absolute(xRange-x))
                        nearestXVal = xRange[xindx] # closest euclidean point
                        if nearestXVal > x or xindx == xdim-1:
                            x0indx = xindx-1; x0 = xRange[x0indx]
                            x1indx = xindx; x1 = xRange[x1indx]
                        elif nearestXVal <= x:
                            x0indx = xindx; x0 = xRange[x0indx]
                            x1indx = xindx+1; x1 = xRange[x1indx]
                        # Then find the bounding y indices for the flow vector
                        yindx = np.argmin(np.absolute(yRange-y))
                        nearestYVal = yRange[yindx] # closest euclidean point
                        if nearestYVal > y or yindx == ydim-1:
                            y0indx = yindx-1; y0 = yRange[y0indx]
                            y1indx = yindx; y1 = yRange[y1indx]
                        elif nearestYVal <= y:
                            y0indx = yindx; y0 = yRange[y0indx]
                            y1indx = yindx+1; y1 = yRange[y1indx]
                        # Now perform rectilinear interpolation to find best vector to
                        # take the inner product with
                        zptsX = xtip[y0indx:y1indx+1,x0indx:x1indx+1]
                        zptsY = ytip[y0indx:y1indx+1,x0indx:x1indx+1]
                        xvec = self.bilinearInterp((x0,x1), (y0,y1), zptsX, x, y)
                        yvec = self.bilinearInterp((x0,x1), (y0,y1), zptsY, x, y)
                        xvecs.append(xvec)
                        yvecs.append(yvec)
                        #print "\t\tFlow at this point = (%f,%f)" % (xvec,yvec)
                        flowNorm = np.sqrt(xvec*xvec+yvec*yvec)
                        #print "\t\t|Flow| at this point = %f" % flowNorm
                        # Do the same with the calcium concentration
                        calciumPts = calData[i,y0indx:y1indx+1,x0indx:x1indx+1]
                        calcium = self.bilinearInterp((x0,x1), (y0,y1), calciumPts, x, y)
                        #print "\t\tCalcium at this point = %f" % calcium
                        #print "\t\tPure Flux through this point = %f" % (np.dot(np.array((xvec,yvec)),slope/slopeNorm))
                        flux += calcium*np.dot(np.array((xvec,yvec)),slope/slopeNorm)
                        #prevFlux = flux
                        #print "\t\tCalcium Adjusted Flux through this point = %f" % (flux-prevFlux)
                    #print "\tTotal Flux at this point = %f" % (flux*ds)
                    totFlux += flux*ds
                pt0 = pt1.copy()
            #print "Total Flux for entire path = %f" % (totFlux)
            #print "numKnots = ", numKnots
            #print "numSteps_dt = ", numSteps_dt
            fluxPts[i] = totFlux/(numKnots*numSteps_dt) # store the average flux along the path
            #print "Average Flux along entire path = %f" % (fluxPts[i])
            #print "TEST: gordon, exit loop";  break
        derivs = np.diff(knots.T,1,1)
        results = {'flux': fluxPts.tolist(), 'dx': derivs[0].tolist(), 'dy': derivs[1].tolist()}

        print "exit calcFluxLinear"
        return results
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    def calcFluxLinear(self, inKnots, calDataIn=None, xtailIn=None, ytailIn=None, xtipsIn=None, ytipsIn=None, width=1, beg=0, end=-1):
        print "enter calcFluxLinear"
        """
        Calculate the flux of a path specified by knot points over a series of
        vector vields determined by xtips and ytips. For subpixel accuracy, the
        path will be linearly interpolated and each vector field will be
        bilinearly interpolated.

        Parameters
        ----------
        inKnots : array-like
            This is the sequence of knot points used to create the piecewise linear interpolant.
            It must be either a tuple, a list, or an ndarray and must be two dimensional of shape
            (2,n) or (n,2) where each point is specified by (x,y).
        calDataIn : array-like
            The actual calcium intensity data. Should have shape (t,m,n) where t is the number of
            time steps and m, n are the rows and columns respectively. The shape should match all
            of the vector information as well.
        xtailIn : ndarray
            A 2 dimensional mesh specifying the x coordinates of the grid for each vector field.
            This must match the shape of ytail and each xtip, ytip. In addition, the x coordinates
            of the knots must be contained within the domain of xtail.
        ytailIn : ndarray
            A 2 dimensional mesh specifying the y coordinates of the grid for each vector field.
            This must match the shape of xtail and each xtip, ytip. In addition, the y coordinates
            of the knots must be contained within the domain of ytail.
        xtipsIn : ndarray
            Each element represents the x component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each ytip
        ytipsIn : ndarray
            Each element represents the y component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each xtip
        width : scalar
            Approximate thickness of process in pixels. Assumed to be 10
        beg : scalar
            Beginning index to slice the time series for analysis
        end : scalar
            Ending index to slice the time series for analysis

        Returns
        -------
        flux : ndarray
            A time series representing the associated flux along the path specified by
            inKnots over the vector fields specified by xtips and ytips.
        """

        if calDataIn is None:
            calData = self.data[beg:end]
        else:
            calData = calDataIn.copy()
        if xtipsIn is None:
            xtips = self.xflow[beg:end]
        else:
            xtips = xtipsIn.copy()
        if ytipsIn is None:
            ytips = self.yflow[beg:end]
        else:
            ytips = ytipsIn.copy()
        if xtailIn is None:
            zdim, ydim, xdim = self.shape
            trash, xtail = np.mgrid[0:ydim, 0:xdim]
        else:
            xtail = xtailIn.copy()
        if ytailIn is None:
            zdim, ydim, xdim = self.shape
            ytail, trash = np.mgrid[0:ydim, 0:xdim]
        else:
            ytail = ytailIn.copy()

        avgImg = self.avgData
        #avgImg[avgImg < np.percentile(avgImg, 60)] = 0
        center = self.calcCentroid(avgImg) # The centroid of the calcium image!
        #print "shape(np.average(calData, axis=0)) = ", ind.shape
        #print "xbar = ", xm
        #print "ybar = ", ym
        knots = self.formatKnots(inKnots)
        width = float(width)
        knots = knots[::len(knots)/50]  # Downsample the number of knots
        numKnots = len(knots)
        # We want the slope to always be pointing toward the soma. This forms
        # the sign convention that positive flux is always inward or towards the
        # soma, and negative flux is always outward or away from the soma.
        # We assume the path was drawn inward (clicked far from soma and dragged
        # toward it). If this is the case then the slope is already toward the soma
        # when calculating along the parameterization direction. Below we test if
        # it was drawn outward. If it was drawn outward we just reverse the order
        # of the knots so that it was drawn inward! Easy fix
        if np.linalg.norm(center-knots[-1]) > np.linalg.norm(center-knots[0]): # Drawn outward
            knots = knots[::-1] # So we reverse the order. Now it is drawn inward
        assert self.verifyData(knots, xtail, ytail), "Interpolation knots must be contained within the vector field!"
        xRange = xtail[0] # Used for finding bounding indices around each point (bilinear interp)
        yRange = ytail[:,0] # Used for finding bounding indices around each point (bilinear interp)
        ydim, xdim = xtail.shape # Used for finding bounding indices around each point (bilinear interp)
        numSteps_dt = 6.0 # Number of points for stepping between the knots points
        numSteps_ds = 5.0 # Number of points for stepping along the line perpendicular to path
        fluxPts = np.zeros(len(xtips))
        xvecs = []
        yvecs = []
        for i,(xtip,ytip) in enumerate(zip(xtips,ytips)):  # For each frame
            d = width/2.0 # integral for flux goes from -d to d
            ds = width/numSteps_ds # The stepsize for above integral
            dt = 1.0/numSteps_dt # stepsize for integral along path
            pt0 = knots[0]
            totFlux = 0
            for j,pt1 in enumerate(knots[1:]):  # For each knot point
                # Linearly interpolating between knot points. Therefore the gradient
                # and slope will not change for the partitioning points along the
                # way between knots
                slope = pt1-pt0 # This is also the derivative
                slopeNorm = np.linalg.norm(slope)
                grad = np.array((-slope[1], slope[0]))/slopeNorm
                #print "\n----------------------------------------"
                #print "Knot# %d" % (j)
                #print "(pt0,pt1) = ((%f,%f),(%f,%f))" % (pt0[0],pt0[1],pt1[0],pt1[1])
                #print "Tangent Vector = (%f,%f)" % (slope[0],slope[1])
                #print "|Tangent Vector| = %f" % (slopeNorm)
                #print "Unit Gradient Vector = (%f,%f)" % (grad[0],grad[1])
                #print "Inner Product between grad and tan vecs = %f" % (np.dot(slope,grad))
                for k,t in enumerate(np.linspace(0,1,numSteps_dt)):  # For each subknot
                    pt = pt0 + slope*t # Step along the line from pt0 to pt1
                    flux = 0
                    #print "\t----------------------------------------"
                    #print "\t0 < t = %f < 1.0 | At step %d out of %d steps" % (t,k,numSteps_dt)
                    #print "\tNext point on path = (%f,%f)" % (pt[0],pt[1])
                    spts = np.arange(-d+ds/2.0,d,ds) # Quadrature points for the midpoint rule:
                    for l,s in enumerate(spts): # For each point along perpendicular line
                        #print "\t\t----------------------------------------"
                        #print "\t\t-d = %f < s = %f < d = %f | At step %d out of %d steps. Stepsize = %f" % (-d,s,d,l,len(spts),ds)
                        x, y = pt + grad*s
                        #print "\t\tNext point tangential to path = (%f,%f)" % (x,y)
                        # At this (x,y), find the bounding indices in xtail and ytail
                        # First find the bounding x indices for the flow vector
                        xindx = np.argmin(np.absolute(xRange-x))
                        nearestXVal = xRange[xindx] # closest euclidean point
                        if nearestXVal > x or xindx == xdim-1:
                            x0indx = xindx-1; x0 = xRange[x0indx]
                            x1indx = xindx; x1 = xRange[x1indx]
                        elif nearestXVal <= x:
                            x0indx = xindx; x0 = xRange[x0indx]
                            x1indx = xindx+1; x1 = xRange[x1indx]
                        # Then find the bounding y indices for the flow vector
                        yindx = np.argmin(np.absolute(yRange-y))
                        nearestYVal = yRange[yindx] # closest euclidean point
                        if nearestYVal > y or yindx == ydim-1:
                            y0indx = yindx-1; y0 = yRange[y0indx]
                            y1indx = yindx; y1 = yRange[y1indx]
                        elif nearestYVal <= y:
                            y0indx = yindx; y0 = yRange[y0indx]
                            y1indx = yindx+1; y1 = yRange[y1indx]
                        # Now perform rectilinear interpolation to find best vector to
                        # take the inner product with
                        zptsX = xtip[y0indx:y1indx+1,x0indx:x1indx+1]
                        zptsY = ytip[y0indx:y1indx+1,x0indx:x1indx+1]
                        xvec = self.bilinearInterp((x0,x1), (y0,y1), zptsX, x, y)
                        yvec = self.bilinearInterp((x0,x1), (y0,y1), zptsY, x, y)
                        xvecs.append(xvec)
                        yvecs.append(yvec)
                        #print "\t\tFlow at this point = (%f,%f)" % (xvec,yvec)
                        flowNorm = np.sqrt(xvec*xvec+yvec*yvec)
                        #print "\t\t|Flow| at this point = %f" % flowNorm
                        # Do the same with the calcium concentration
                        calciumPts = calData[i,y0indx:y1indx+1,x0indx:x1indx+1]
                        calcium = self.bilinearInterp((x0,x1), (y0,y1), calciumPts, x, y)
                        #print "\t\tCalcium at this point = %f" % calcium
                        #print "\t\tPure Flux through this point = %f" % (np.dot(np.array((xvec,yvec)),slope/slopeNorm))
                        flux += calcium*np.dot(np.array((xvec,yvec)),slope/slopeNorm)
                        #prevFlux = flux
                        #print "\t\tCalcium Adjusted Flux through this point = %f" % (flux-prevFlux)
                    #print "\tTotal Flux at this point = %f" % (flux*ds)
                    totFlux += flux*ds
                pt0 = pt1.copy()
            #print "Total Flux for entire path = %f" % (totFlux)
            #print "numKnots = ", numKnots
            #print "numSteps_dt = ", numSteps_dt
            fluxPts[i] = totFlux/(numKnots*numSteps_dt) # store the average flux along the path
            #print "Average Flux along entire path = %f" % (fluxPts[i])
        derivs = np.diff(knots.T,1,1)
        results = {'flux': fluxPts.tolist(), 'dx': derivs[0].tolist(), 'dy': derivs[1].tolist()}

        print "exit calcFluxLinear"
        return results
    #----------------------------------------------------------------------

    def calcPseudofluxLinear(self, inKnots, xtail, ytail, xtips, ytips):
        """
        Calculate the 'pseudoflux' of a path specified by knot points over a series of
        vector vields determined by xtips and ytips. The path will be linearly interpolated
        and each vector field will be bilinearly interpolated.

        Parameters
        ----------
        inKnots : array-like
            This is the sequence of knot points used to create the piecewise linear interpolant.
            It must be either a tuple, a list, or an ndarray and must be two dimensional of shape
            (2,n) or (n,2) where each point is specified by (x,y).
        xtail : ndarray
            A 2 dimensional mesh specifying the x coordinates of the grid for each vector field.
            This must match the shape of ytail and each xtip, ytip. In addition, the x coordinates
            of the knots must be contained within the range of xtail.
        ytail : ndarray
            A 2 dimensional mesh specifying the y coordinates of the grid for each vector field.
            This must match the shape of xtail and each xtip, ytip. In addition, the y coordinates
            of the knots must be contained within the range of ytail.
        xtips : ndarray
            Each element represents the x component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each ytip
        ytips : ndarray
            Each element represents the y component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each xtip

        Returns
        -------
        pFlux : ndarray

            A time series representing the associated 'pseudoflux' along the path specified by
            inKnots over the vector fields specified by xtips and ytips.

        """

        knots = self.formatKnots(inKnots)
        assert verifyData(knots, xtail, ytail), "Interpolation knots must be contained within the vector field!"
        xRange = xtail[0] # Used for finding bounding indices around each point (bilinear interp)
        yRange = ytail[:,0] # Used for finding bounding indices around each point (bilinear interp)
        ydim, xdim = xtail.shape # Used for finding bounding indices around each point (bilinear interp)
        numKnots = len(knots)
        numSteps = 10
        pFlux = np.zeros(len(xtips))
        for i,(xtip,ytip) in enumerate(zip(xtips,ytips)):
            tot = 0
            pt0 = knots[0]
            for pt1 in knots[1:]:
                slope = pt1-pt0 # This is also the derivative
                dt = 1.0/numSteps
                totXvec = 0; totYvec = 0; totMag = 0;
                for t in np.arange(dt/2.0,1,dt): # Quadrature points for the midpoint rule
                    x, y = pt0 + slope*t # Step along the line from pt0 to pt1

                    # At this (x,y), find the bounding indices in xtail and ytail
                    # First find the bounding x indices for the flow vector
                    xindx = np.argmin(np.absolute(xRange-x))
                    nearestXVal = xRange[xindx] # closest euclidean point
                    if nearestXVal > x or xindx == xdim-1:
                        x0indx = xindx-1; x0 = xRange[x0indx]
                        x1indx = xindx; x1 = xRange[x1indx]
                    elif nearestXVal <= x:
                        x0indx = xindx; x0 = xRange[x0indx]
                        x1indx = xindx+1; x1 = xRange[x1indx]
                    # Then find the bounding y indices for the flow vector
                    yindx = np.argmin(np.absolute(yRange-y))
                    nearestYVal = yRange[yindx] # closest euclidean point
                    if nearestYVal > y or yindx == ydim-1:
                        y0indx = yindx-1; y0 = yRange[y0indx]
                        y1indx = yindx; y1 = yRange[y1indx]
                    elif nearestYVal <= y:
                        y0indx = yindx; y0 = yRange[y0indx]
                        y1indx = yindx+1; y1 = yRange[y1indx]
                    # Now perform rectilinear interpolation to find best vector to
                    # take the inner product with
                    zptsX = xtip[y0indx:y1indx+1,x0indx:x1indx+1]
                    zptsY = ytip[y0indx:y1indx+1,x0indx:x1indx+1]
                    xvec = bilinearInterp((x0,x1), (y0,y1), zptsX, x, y)
                    yvec = bilinearInterp((x0,x1), (y0,y1), zptsY, x, y)
                    totXvec += xvec
                    totYvec += yvec
                    totMag += np.sqrt(xvec*xvec+yvec*yvec)
                tot += slope[0]*totXvec+slope[1]*totYvec
                pt0 = pt1
            pFlux[i] = tot*dt

        return pFlux

    def calcPseudofluxPpchip(self, inKnots, xtail, ytail, xtips, ytips):
        """
        Calculate the 'pseudoflux' of a path specified by knot points over a series of
        vector vields determined by xtips and ytips. The path will be interpolated by
        a parametric piecewise cubic hermite interpolant and each vector field will
        be bilinearly interpolated.

        Parameters
        ----------
        inKnots : array-like
            This is the sequence of knot points used to create the hermite spline.
            It must be either a tuple, a list, or an ndarray and must be two
            dimensional of shape (2,n) or (n,2) where each point is specified by
            (x,y).
        xtail : ndarray
            A 2 dimensional mesh specifying the x coordinates of the grid for each vector field.
            This must match the shape of ytail and each xtip, ytip. In addition, the x coordinates
            of the knots must be contained within the range of xtail.
        ytail : ndarray
            A 2 dimensional mesh specifying the y coordinates of the grid for each vector field.
            This must match the shape of xtail and each xtip, ytip. In addition, the y coordinates
            of the knots must be contained within the range of ytail.
        xtips : ndarray
            Each element represents the x component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each ytip
        ytips : ndarray
            Each element represents the y component of the vector associated with the corresponding
            spatial (x,y) grid point found in xtail and ytail. Must be 3 dimensional where the first
            dimension represents time and the remaining spatial dimensions must match xtail, ytail
            and each xtip

        Returns
        -------
        (pFlux, normPflux) : (ndarray, ndarray)
            Each time series represents the associated 'pseudoflux' along the path specified by
            inKnots over the vector fields specified by xtips and ytips. The latter of the two
            is normalized between -1 and 1.

        Notes
        -----
            - In the future only return one normalized array. For now I return both
            for comparison purposes.
        """

        verbose = 0
        knots = self.formatKnots(inKnots)
        assert verifyData(knots, xtail, ytail), "Interpolation knots must be contained within the vector field!"
        # Create the interpolant and get all the points and derivatives before looping
        ppchip = ParaPCHIP(knots)
        numPts = 100
        dt = ppchip.length/numPts
        samplePts = np.arange(dt/2,ppchip.length,dt)
        pts, derivs = ppchip(samplePts, d=1)
        xRange = xtail[0]
        yRange = ytail[:,0]
        ydim, xdim = xtail.shape
        pFlux = np.zeros(len(xtips))
        normPflux = np.zeros(len(xtips))
        for i,(xtip,ytip) in enumerate(zip(xtips,ytips)):
            tot = 0
            for (x,y),(dx,dy) in zip(pts.T,derivs.T):
                if verbose:
                    print "-----------------------"
                    print "(x,y) = (%f,%f)" % (x,y)

                # First find the vector from the flow field corresponding to this point
                # Find the bounding x indices for the vector tip
                xindx = np.argmin(np.absolute(xRange-x))
                nearestXVal = xRange[xindx] # closest euclidean point
                if verbose:
                    print "xRange = ", xRange
                    print "xindx = ", xindx
                    print "nearestXVal = ", nearestXVal
                if nearestXVal > x or xindx == xdim-1:
                    x0indx = xindx-1; x0 = xRange[x0indx]
                    x1indx = xindx; x1 = xRange[x1indx]
                elif nearestXVal <= x:
                    x0indx = xindx; x0 = xRange[x0indx]
                    x1indx = xindx+1; x1 = xRange[x1indx]
                if verbose:
                    print "X values at (%f,%f) = (%f,%f)" % (x0indx,x1indx,x0,x1)
                # Find the bounding y indices for the vector
                yindx = np.argmin(np.absolute(yRange-y))
                nearestYVal = yRange[yindx] # closest euclidean point
                if verbose:
                    print "yRange = ", yRange
                    print "yindx = ", yindx
                    print "nearestYVal = ", nearestYVal
                if nearestYVal > y or yindx == ydim-1:
                    y0indx = yindx-1; y0 = yRange[y0indx]
                    y1indx = yindx; y1 = yRange[y1indx]
                elif nearestYVal <= y:
                    y0indx = yindx; y0 = yRange[y0indx]
                    y1indx = yindx+1; y1 = yRange[y1indx]
                if verbose:
                    print "Y values at (%f,%f) = (%f,%f)" % (y0indx,y1indx,y0,y1)

                # Now perform rectilinear interpolation to find best vector to
                # take the inner product with
                xpts = (x0,x1)
                ypts = (y0,y1)
                zptsX = xtip[y0indx:y1indx+1,x0indx:x1indx+1]
                zptsY = ytip[y0indx:y1indx+1,x0indx:x1indx+1]
                if verbose:
                    print "xpts = ", xpts
                    print "ypts = ", ypts
                    print "--zptsX--\n", zptsX
                    print "--zptsY--\n", zptsY
                xvec = bilinearInterp(xpts, ypts, zptsX, x, y)
                yvec = bilinearInterp(xpts, ypts, zptsY, x, y)
                flowNorm = np.sqrt(xvec*xvec+yvec*yvec)
                if verbose:
                    print "(xvec,yvec) at (%f,%f) = (%f,%f)" % (x0,y0,xtip[y0indx,x0indx],ytip[y0indx,x0indx])
                    print "(xvec,yvec) at (%f,%f) = (%f,%f)" % (x0,y1,xtip[y1indx,x0indx],ytip[y1indx,x0indx])
                    print "(xvec,yvec) at (%f,%f) = (%f,%f)" % (x1,y0,xtip[y0indx,x1indx],ytip[y0indx,x1indx])
                    print "(xvec,yvec) at (%f,%f) = (%f,%f)" % (x1,y1,xtip[y1indx,x1indx],ytip[y1indx,x1indx])
                    print "interpolated (xvec,yvec) at (%f,%f) = (%f,%f)" % (x,y,xvec,yvec)
                if verbose:
                    print "Derivative at (%f,%f) = (%f,%f)" % (x,y,xp,yp)
                    print "Inner product of Deriv and Interp Vec = %f" % (np.dot(np.array((xvec,yvec)), np.array((xp,yp))))
                tot += np.dot(np.array((xvec,yvec)), np.array((dx,dy)))
            if verbose:
                print "totFlow = ", totFlow
                if normFlow:
                    print "totMag = ", totMag
            pFlux[i] = totNum*dt
            normPflux[i] = totNum/totDen

        return (pFlux, normPflux)

    def calcPseudofluxPix(self, xVecs, yVecs, path, norm=False):
        """
        Sloppily calculate the pseudoflux just by going from one pixel to the next
        """

        pseudoFlux = np.zeros(len(xVecs))
        for i in range(len(xVecs)):
            prevPt = path[0]
            normalizer = 0 if norm else 1.0
            for pt in path[1:]:
                y0 = prevPt[1]; x0 = prevPt[0]
                vec1 = pt-prevPt
                vec2 = np.array((xVecs[i,y0,x0],yVecs[i,y0,x0]))
                pseudoFlux[i] += np.dot(vec1, vec2)
                if norm:
                    normalizer += np.linalg.norm(vec1)*np.linalg.norm(vec2)
                prevPt = pt
            pseudoFlux[i] /= normalizer

        return pseudoFlux
