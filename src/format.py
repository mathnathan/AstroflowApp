import numpy as np

knots = np.loadtxt("knotPoints.txt")
print "knots.shape = ", knots.shape

with open("knotPoints.csv", 'w') as fd:
    fd.write("[[")
    for v in knots.T[0,:-1]:
        fd.write("%f, " % (v))
    fd.write("%f], [" % (knots.T[0,-1]))
    for v in knots.T[1,:-1]:
        fd.write("%f, " % (v))
    fd.write("%f]]" % (knots.T[1,-1]))
