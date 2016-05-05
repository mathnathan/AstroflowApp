from flask import Flask, render_template, request, json
import sys
sys.path.insert(0, "./src")
import hotflux as hf
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/calcFlux', methods = ['POST'])
def calcFlux():
    """
    This function expects the path to be passed as a JSON object
    in the following form
    { path: [[xpt1, xpt2, ...], [ypt1, ypt2, ...]] }
    """

    print "\nIn calcFlux!!\n"
    jsonData = request.get_json()
    path = np.array(jsonData["path"])
    flux = analyzer.calcFluxLinear(path)
    return json.jsonify({"flux": flux.tolist()})


@app.route('/findHotspots', methods = ['POST'])
def findHotspots():
    """
    This function expects the path to be passed as a JSON object
    in the following form
    { path: [[xpt1, xpt2, ...], [ypt1, ypt2, ...]] }
    """

    print "\nIn findHotspots!!\n"
    jsonData = request.get_json()
    path = np.array(jsonData["path"])
    hotspots, flows = analyzer.findHotspots(path)
    return json.jsonify({"hotspots": hotspots, "flows": flows.tolist()})

if __name__ == '__main__':

    if len(sys.argv) == 2:
        dataPath = sys.argv[1]
    else:
        dataPath = "/home/ndc08/code/research/compneuro/max_planck_jupiter/nathans_project1/data/TSeries-06232015-1045__EpOri(12sec_2dir)_Site2_4Hz_0.75ISO_Astro3_AL/TSeries-06232015-1045__EpOri(12sec_2dir)_Site2_4Hz_0.75ISO_Astro3_AL.tif"
    analyzer = hf.HotFlux(dataPath)
    app.run(debug=True)
