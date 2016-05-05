import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np


def pltRoiTimeSeries(data, boxes, times):
    """Non-interactive. Specify a bounding box"""

    series = []
    rois = zip(times, boxes)
    for roi in rois:
        beg = roi[0][0]
        end = roi[0][1]
        x0 = roi[1][0]
        y0 = roi[1][1]
        x1 = roi[1][2]
        y1 = roi[1][3]
        si = np.ndarray((end-beg))
        for time, frame in enumerate(data[beg:end]):
            pixels = frame[y0:y1, x0:x1]
            si[time] = np.average(pixels)
        series.append(si)

    import random

    maxAmp = np.max([np.max(s) for s in series])
    minAmp = np.min([np.min(s) for s in series])
    avg = np.average(data, axis=0)
    numSeries = len(series)
    plt.subplot(numSeries+1, 1, 1)
    plt.imshow(avg)
    plt.title("ROI, Time Series")
    for indx in range(numSeries):
        r = lambda: random.randint(0, 255)
        clr = '#%02X%02X%02X' % (r(), r(), r())
        ax1 = plt.subplot(numSeries+1, 1, 1)
        x0, y0, x1, y1 = boxes[indx]
        ax1.add_patch(patches.Rectangle((
            x0, y0), x1-x0, y1-y0, fill=0, edgecolor=clr))  # draw square
        plt.subplot(numSeries+1, 1, indx+2)
        timeArray = np.arange(times[indx][0], times[indx][1])
        plt.plot(timeArray, series[indx], clr)
        plt.xlabel("Time")
        plt.ylabel("Avg Intensity")
        plt.ylim([minAmp, maxAmp])
    plt.show()

    return series


def plotFlow(X, Y, U, V):

    print "-- Plotting Flow..."

    if False:
        speed = np.sqrt(U*U + V*V)

        plt.figure()
        plt.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
        plt.colorbar()

        f, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.streamplot(X, Y, U, V, density=[0.5, 1])

        lw = 5*speed/speed.max()
        ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)


def animateData(data, numFrames=10, speed=50):

    import matplotlib.animation as animation

    print "-- Animating Data..."

    fig = plt.figure(plt.gcf().number+1)
    # The update function is called between every frame
    update = lambda num: [plt.imshow(data[num])]
    # The LHS animation object must exist for animation to work
    img_ani = animation.FuncAnimation(fig, update, numFrames, interval=speed,
                                      blit=True)
    # plt.show()
    return img_ani


def plotDiscreteSurface(*args):
    """ Each element is a tuple containing the x, y, and z components of the
    data point in the plot. i.e. [(x1,y1,z1), (x2,y2,z2), ..., (xn,yn,zy)]. They
    should all be of shape (m,n). Such as those made from meshgrid"""

    print "-- Plotting 3D Discrete Surfaces..."
    from mpl_toolkits.mplot3d import axes3d

    for data in args:
        X = data[0]
        Xmax = X.max()
        Xmin = X.min()
        Xperc = (Xmax-Xmin)*0.15
        Y = data[1]
        Ymax = Y.max()
        Ymin = Y.min()
        Yperc = (Ymax-Ymin)*0.15
        Z = data[2]
        print "\n Z = ", Z, "\n"
        Zmax = Z.max()
        Zmin = Z.min()
        Zperc = (Zmax-Zmin)*0.15
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.3)
        cset = ax.contour(
            X, Y, Z, zdir='z', offset=Zmin-Zperc, cmap=plt.cm.coolwarm)
        cset = ax.contour(
            X, Y, Z, zdir='x', offset=Xmax+Xperc, cmap=plt.cm.coolwarm)
        cset = ax.contour(
            X, Y, Z, zdir='y', offset=Ymax+Yperc, cmap=plt.cm.coolwarm)

        ax.set_xlabel('Space')
        ax.set_xlim(Xmin, Xmax)
        ax.set_ylabel('Time')
        ax.set_ylim(Ymin, Ymax)
        ax.set_zlabel('Calcium Flux')
        ax.set_zlim(Zmin, Zmax)
        w = data[3]
        ax.set_title("Flux with Width = %0.1f" % (w))


def plotContinuousSurface(func, xlims, ylims, numPts=1000.0):

    print "-- Plotting 3D Continuous Surfaces..."

    x0 = xlims[0]
    x1 = xlims[1]
    xstep = (x1-x0)/numPts
    y0 = ylims[0]
    y1 = ylims[1]
    ystep = (y1-y0)/numPts
    xpts = np.arange(x0, x1, xstep)
    ypts = np.arange(y0, y1, ystep)
    Z = func(xpts, ypts, grid=True)
    Y, X = np.meshgrid(xpts, ypts)
    plotDiscreteSurface((X, Y, Z))


def roiSelector(data, thresh):
    """ data is a 3 dimensional array. A plot will appear with the average of
        data and a time series underneath. Use the mouse to draw a closed loop around
        the pixels you would like for the roi. Those pixels will be highlighted
        and the corresponding time series will appear in the plot underneath."""

    zdim, ydim, xdim = data.shape
    avg = np.average(data, axis=0)

    #plt.ion()
    fig = plt.figure()
    #fig.set_figheight(11)
    #fig.set_figwidth(8.5)
    #fig.patch.set_facecolor('none')
    img = plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
    im = img.imshow(avg)
    sig = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
    sigLines = sig.plot(np.zeros(zdim), lw=1.2)[0]
    sigThresh = sig.plot(np.zeros(zdim), color='k', ls='dashed', lw=0.7)[0]
    sig.set_xlim([0, zdim])
    sig.set_xlabel("Time")
    sig.set_title("ROI Time Series")
    ypts, xpts = np.mgrid[0:ydim:1, 0:xdim:1]
    pts = zip(xpts.flatten(), ypts.flatten())
    plt.tight_layout()
    tSeries = np.ndarray(len(data))
    def select(verts):
        cpy = im.to_rgba(avg, alpha=0.7)
        roiPixels = Path(verts).contains_points(pts).reshape(ydim, xdim)
        cpy[roiPixels, -1] = 1.0 # Now only the roi is fully brightened
        im.set_data(cpy)
        tSeries[:] = np.asarray([np.average(roi) for roi in data[:, roiPixels]]).copy()
        sigLines.set_ydata(tSeries)
        yvals = np.ones(tSeries.size)*(thresh*tSeries.max())
        sigThresh.set_ydata(yvals)
        miny = np.min(tSeries)
        maxy = np.max(tSeries)
        rng = np.abs(maxy-miny)
        sig.set_ylim([miny-rng*0.1, maxy+rng*0.1])
        im.figure.canvas.draw_idle()
        sig.figure.canvas.draw_idle()

    lasso = LassoSelector(img, select, lineprops={'color': 'black'})
    plt.draw()

    raw_input('Press any key to stop ROI session')
    lasso.disconnect_events()
    #plt.ioff()
    plt.close()

    return tSeries
