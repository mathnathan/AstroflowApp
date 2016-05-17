import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

numKnots = 12
numSamples = 100

xpts = np.linspace(0,2*np.pi,numKnots)
ypts = np.sin(xpts)

plt.scatter(xpts, ypts, color='k', label="knots")

tck, u = interpolate.splprep([xpts, ypts], s=0.0)
sampleParams = np.linspace(0,1,numSamples)
samplePts = np.array(interpolate.splev(sampleParams, tck, der=0))

plt.plot(samplePts[0], samplePts[1], color='b', label='sin')

derivs = np.array(interpolate.splev(sampleParams, tck, der=1))
normDerivs = derivs / np.linalg.norm(derivs, axis=0) # Normalize the derivative

plt.plot(samplePts[0], derivs[1], color='r', label='deriv')
plt.plot(samplePts[0], normDerivs[1], color='g', label='norm deriv')
plt.legend()
plt.show()
