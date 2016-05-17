import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

numKnots = 12
numSamples = 1000

xpts = np.linspace(0,2*np.pi,numKnots)
ypts = np.sin(xpts)
exact_deriv = np.cos(xpts)

plt.scatter(xpts, ypts, color='k', label="knots")

tck, u = interpolate.splprep([xpts, ypts], s=0.0)
sampleParams = np.linspace(0,1,numSamples)
samplePts = np.array(interpolate.splev(sampleParams, tck, der=0))

plt.plot(samplePts[0], samplePts[1], color='b', label='sin')

# Finite-difference derivative based on the sample points
# dy/dx = (dy/dp) / (dx/dp)
x = samplePts[0]
y = samplePts[1]
derivs = np.zeros(len(x))
derivs[1:-1] = (y[2:] - y[0:-2]) / (x[2:] - x[0:-2])
derivs[0]    = (y[1] - y[0]) / (x[1] - x[0])
derivs[-1]   = (y[-1] - y[-2]) / (x[-1] - x[-2])

plt.plot(x, derivs, color='cyan', label='F-D deriv')
plt.scatter(xpts, exact_deriv, color='black', label='exact deriv')

derivs = np.array(interpolate.splev(sampleParams, tck, der=1))
print derivs
normDerivs = derivs / np.linalg.norm(derivs, axis=0) # Normalize the derivative

plt.plot(samplePts[0], derivs[1]/derivs[0], color='r', label='deriv')
#plt.plot(samplePts[0], normDerivs[1], color='g', label='norm deriv')
plt.legend()
plt.show()
