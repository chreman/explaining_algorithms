"""

"""

import sys
import logging
from flask import Flask, render_template, redirect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from keras.applications import inception_v3 as inc_net

from creditscoring import CSModel
from petimages import NNModel

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("applogger")
sh_out = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] '
                              '[%(levelname)s] %(message)s')
sh_out.setFormatter(formatter)
logger.addHandler(sh_out)

logger.info("Creating credit score model.")
application = Flask('explain_algorithms')
csmodel = CSModel(logger=logger, datapath="data/creditscoring")
csmodel.preprocess()
csmodel.create_model()
csmodel.create_model_explainer()

logger.info("Creating neural net image model.")
inet_model = inc_net.InceptionV3()
nnmodel = NNModel(inet_model, logger=logger,
                  datapath="data/oxfordiiipets")
nnmodel.preprocess()
logger.info("Completed NNModel preprocessing.")
nnmodel.create_model_explainer()


@app.route('/')
def main():
    return redirect('/index')


@app.route("/index")
def index():
    return render_template(
        "index.html"
    )


@app.route("/creditscoring")
def render_creditscoring():
    logger.info("Creating random credit score examples.")
    randoms = [np.random.randint(0, csmodel.test.shape[0]) for i in range(4)]
    df = pd.DataFrame(csmodel.test[randoms],
                      columns=csmodel.feature_names)
    df = csmodel.remap_categoricals(df)
    random_exps = []
    true_labels = []
    for r in randoms:
        random_exp, true_label = csmodel.get_explanation(r)
        random_exps.append(random_exp)
        if true_label == 0:
            true_labels.append("Not Credit Worthy")
        else:
            true_labels.append("Credit Worthy")
    df["True Labels"] = true_labels
    table = df.to_html(classes="table table-striped .table-condensed")
    logger.info("Rendering random credit score examples.")
    return render_template(
        "creditscoring.html",
        table=table,
        random_exps=random_exps
    )


@app.route("/petimages")
def render_petimages():
    logger.info("Creating random petimage examples.")
    randoms = [np.random.randint(0, len(nnmodel.images)) for i in range(3)]
    random_exps = []
    for r in randoms:
        random_exps.append(nnmodel.get_explanation(r))
    logger.info("Rendering petimage examples.")
    return render_template(
        "petimages.html",
        random_exps=random_exps
    )


if __name__ == '__main__':
    application.run(debug=True)
