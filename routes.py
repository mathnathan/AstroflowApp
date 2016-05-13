from flask import Flask, render_template, request, json
import sys
sys.path.insert(0, "./src")
import hotflux as hf
import numpy as np

app = Flask(__name__)


#@app.route('/')
#def home():
#    return render_template('home.html')


#@app.route('/about')
#def about():
#    return render_template('about.html')


@app.route('/getMetadata', methods=['GET'])
def getMetadata():
    zdim, ydim, xdim = analyzer.data.shape
    return json.jsonify({ "frames": zdim, "ydim": ydim, "xdim": xdim })


@app.route('/getFlow', methods=['GET'])
def getFlow():
    frameNum = int(request.args.get('i'))
    xflow = analyzer.xflow[frameNum]
    yflow = analyzer.yflow[frameNum]
    print "got flows at %d" % (frameNum)
    return json.jsonify({ "xflow" : xflow.tolist(), "yflow" : yflow.tolist() })


@app.route('/getFrame', methods=['GET'])
def getFrame():
    frameNum = int(request.args.get('i'))
    frame = analyzer.data[frameNum]
    print "got frame%d" % (frameNum)
    return json.jsonify({ "frame" : frame.tolist() })


@app.route('/calcFlux', methods=['POST'])
def calcFlux():
    """
    This function expects the path to be passed as a JSON object
    in the following form
    { "path": [[xpt1, xpt2, ...], [ypt1, ypt2, ...]] }
    """

    print "\nIn calcFlux!!\n"
    jsonData = request.get_json()
    path = np.array(jsonData["path"])
    flux = analyzer.calcFluxLinear(path)
    return json.jsonify({"flux": flux.tolist()})


@app.route('/findHotspots', methods=['POST'])
def findHotspots():
    """
    This function expects the path to be passed as a JSON object
    in the following form
    { "path": [[xpt1, xpt2, ...], [ypt1, ypt2, ...]] }
    """

    print "\nIn findHotspots!!\n"
    jsonData = request.get_json()
    path = np.array(jsonData["path"])
    hotspots, flows = analyzer.findHotspots(path)
    return json.jsonify({"hotspots": hotspots, "flows": flows.tolist()})

if __name__ == '__main__':

    print "sys.argv = ", sys.argv
    dataPath = sys.argv[-1]
    print "dataPath = ", dataPath
    analyzer = hf.HotFlux(dataPath)
    app.run(debug=True)
