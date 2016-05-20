from flask import Flask, render_template, request, json
import sys
sys.path.insert(0, "./src")
import hotflux as hf
import numpy as np

app = Flask(__name__)


@app.route('/suicide', methods=['POST'])
def suicide():
    import thread
    thread.interrupt_main()  # Try this first. Should gracefully exist all threads
    import os
    os._exit()  # This is emergency backup. If above fails, hard exit the program


@app.route('/getAverage', methods=['GET'])
def getAverage():
    start, stop = getBounds(request.args)
    if start == 0 and stop == -1:
        avg = analyzer.avgData
    else:
        avg = np.average(analyzer.data[start:stop], 0)
    return json.jsonify({ "average": avg.tolist() })


@app.route('/getMetadata', methods=['GET'])
def getMetadata():
    zdim, ydim, xdim = analyzer.data.shape
    return json.jsonify({ "frames": zdim, "ydim": ydim, "xdim": xdim })


@app.route('/getFlow', methods=['GET'])
def getFlow():
    if 'i' in request.args:
        frameNum = int(request.args['i'])
        xflow = analyzer.xflow[frameNum]
        yflow = analyzer.yflow[frameNum]
    else:
        start, stop = getBounds(request.args)
        xflow = analyzer.xflow[start:stop]
        yflow = analyzer.yflow[start:stop]
    return json.jsonify({ "xflow" : xflow.tolist(), "yflow" : yflow.tolist() })


@app.route('/getFrame', methods=['GET'])
def getFrame():
    if 'i' in request.args:
        frameNum = int(request.args['i'])
        frame = analyzer.data[frameNum]
    else:
        start, stop = getBounds(request.args)
        frame = analyzer.data[start:stop]
    return json.jsonify({ "frame" : frame.tolist() })


@app.route('/calcFlux', methods=['POST'])
def calcFlux():
    """
    This function expects the path to be passed as a JSON object
    in the following form
    { "path": [[xpt1, xpt2, ...], [ypt1, ypt2, ...]] }
    """

    jsonData = request.get_json()
    path = np.array(jsonData["path"])
    if 'i' in jsonData:
        start = jsonData['i']
        stop = start + 1
    else:
        start, stop = getBounds(jsonData)

    #results = {'flux': fluxPts.tolist(), 'dx': derivs[0].tolist(), 'dy': derivs[1].tolist()}
    (flux, dx, dy) = analyzer.calcFlux(path, beg=start, end=stop)
    return json.jsonify({"flux": flux, "dx": dx, "dy": dy})


@app.route('/findHotspots', methods=['POST'])
def findHotspots():
    """
    This function expects the path to be passed as a JSON object
    in the following form
    { "path": [[xpt1, xpt2, ...], [ypt1, ypt2, ...]] }
    """

    jsonData = request.get_json()
    path = np.array(jsonData["path"])
    if 'i' in jsonData:
        start = jsonData['i']
        stop = start + 1
    else:
        start, stop = getBounds(jsonData)

    (hotspots, pathPts, xflow, yflow) = analyzer.findHotspots(path, beg=start, end=stop)
    return json.jsonify({'hotspots': hotspots, 'pathPts': pathPts, 'xflow': xflow, 'yflow': yflow})

def getBounds(request):
    start = 0; stop = -1
    if 'beg' in request:
        start = int(request['beg'])
        #assert start >= 0 and start < analyzer.data.shape[0]-1, "The starting index must be in [%d,%d]" % (0,analyzer.data.shape[0]-2)
    if 'end' in request:
        stop = int(request['end'])
        #assert stop >= 1 and stop < analyzer.data.shape[0], "The ending index must be in [%d,%d]" % (1,analyzer.data.shape[0]-1)
    #assert stop > start, "The ending index, %d, must be greater than the starting index, %d." % (stop, start)

    return (start, stop)

if __name__ == '__main__':

    print "sys.argv = ", sys.argv
    dataPath = sys.argv[-1]
    print "dataPath = ", dataPath
    analyzer = hf.HotFlux(dataPath)
    app.run(debug=True)
