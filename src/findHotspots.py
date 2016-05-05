import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys
from matplotlib.widgets import LassoSelector
import random
from IPython import embed
import hotflux as hf

#tests = ["centroid"]
tests = []

plt.ion()

dataPath = "/home/ndc08/code/research/compneuro/max_planck_jupiter/nathans_project1/data/TSeries-06232015-1045__EpOri(12sec_2dir)_Site2_4Hz_0.75ISO_Astro3_AL/TSeries-06232015-1045__EpOri(12sec_2dir)_Site2_4Hz_0.75ISO_Astro3_AL.tif"
#dataPath = "data/synthetic_flow/diffusive_event.tif"
analyzer = hf.HotFlux(dataPath)

if "centroid" in tests:
    plt.imshow(analyzer.avgData)
    centr = analyzer.centroid()
    print "centr = ", centr
    plt.scatter([centr[0]], [centr[1]], s=30)
    plt.show()

zdim, ydim, xdim = analyzer.shape
#blurData = gauss(analyzer.data, (0.9,0.65,0.65))
blurData = analyzer.data

print "blurData.shape = ", blurData.shape

# Now prepare the interactive plot to visualize the pseudoflux calculations
mainImg = plt.figure()
img = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
im = img.imshow(analyzer.avgData) # This is where you will select the path
plots = plt.figure()
hotspots = []
numTimes = 5
for hs in range(numTimes):
    hotspots.append(plt.subplot2grid((numTimes, 1), (hs, 0), rowspan=1, colspan=1))
#r = lambda: random.randint(0, 255)
#clr = '#%02X%02X%02X' % (r(), r(), r())

ytail, xtail = np.mgrid[0:ydim:1, 0:xdim:1]
pts = zip(xtail.flatten(), ytail.flatten())

def select(verts):
    cpy = im.to_rgba(analyzer.avgData)
    verts = np.asarray(verts)[:-1] # For some reason the last point is repeated
    numKnots = 100
    xpts = np.arange(numKnots)
    ypts = np.zeros(numKnots)
    lines = []
    for hs in hotspots:
        hs.plot(xpts, ypts, ls='dashed')[0]
        lines.append(hs.plot(xpts, ypts, color='b', lw=1.5)[0])

    # First we find the pixels coordinates corresponding to the path drawn
    path = np.int32(np.round(verts)).tolist()
    pathPixels = [] # Pixel coordinates are stored in here
    for pix in path:
        if pix not in pathPixels:
            pathPixels.append(pix)

    # Now that we have the pixel locations where the path was drawn, lets color
    # the path so the user can see the path they drew
    xlocs, ylocs = np.array(pathPixels).T.tolist()
    cpy[ylocs, xlocs, :-1] = 0 # Set the (R,G,B) channels to zero i.e. black
    im.set_data(cpy)
    im.figure.canvas.draw_idle()

    # Calculate the flux along this path using the flow determined above
    miny = 100000; maxy = -100000
    pts, flowVtime = analyzer.findHotspots(verts, analyzer.xflow[191:196], analyzer.yflow[191:196], numKnots)
    #pts, flowVtime = analyzer.findHotspots(verts, analyzer.xflow, analyzer.yflow, numKnots)
    # Now just update the plots with the new data!
    for i,(line,axis) in enumerate(zip(lines,hotspots)):
        #line.set_xdata(pts[0])
        line.set_ydata(flowVtime[i])
        m1 = np.min(flowVtime[i])
        miny = m1 if m1 < miny else miny
        m2 = np.max(flowVtime[i])
        maxy = m2 if m2 > maxy else maxy

    for axis in hotspots:
        axis.set_ylim([1.1*miny, 1.1*maxy])
        axis.figure.canvas.draw_idle()

lasso = LassoSelector(img, select, lineprops={'color': 'black'})
plt.draw()
raw_input('Press any key to stop ROI session')

lasso.disconnect_events()
plt.ioff()
plt.close()

sys.exit()
