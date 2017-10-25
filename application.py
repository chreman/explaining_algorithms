"""

"""

import sys
import logging
from flask import Flask, render_template, redirect, url_for, request
import numpy as np
import pandas as pd
import matplotlib
from keras.applications import inception_v3 as inc_net
from keras.applications import resnet50

from creditscoring import CSModel
from petimages import NNModel
from detox import TextModel

matplotlib.use('Agg')

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("applogger")
sh_out = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] '
                              '[%(levelname)s] %(message)s')
sh_out.setFormatter(formatter)
logger.addHandler(sh_out)

application = Flask('explain_algorithms')

logger.info("Creating credit score model.")
csmodel = CSModel(logger=logger, datapath="data/creditscoring")
csmodel.preprocess()
csmodel.create_model()
csmodel.create_model_explainer()
logger.info("Created credit score model.")

logger.info("Creating neural net image model.")
inet_model = inc_net.InceptionV3()
# inet_model = resnet50.ResNet50()
nnmodel = NNModel(inet_model, logger=logger,
                  datapath="data/oxfordiiipets")
nnmodel.preprocess()
logger.info("Completed NNModel preprocessing.")
nnmodel.create_model_explainer()
logger.info("Created neural net image model.")

logger.info("Creating text classification model.")
tmodel = TextModel(logger=logger, datapath="data/WMTalk/toxicity/")
tmodel.preprocess()
tmodel.create_model()
tmodel.create_model_explainer()
logger.info("Created text classification model.")


@application.route('/')
def main():
    return redirect('/index')


@application.route("/index")
def index():
    return render_template(
        "index.html"
    )


@application.route("/creditscoring", methods=['GET', 'POST'])
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
    df["True Label"] = true_labels
    table = df.to_html(classes="table table-striped .table-condensed")
    logger.info("Rendering random credit score examples.")
    return render_template(
        "creditscoring.html",
        table=table,
        random_exps=random_exps
    )

@application.route("/creditscoring_input", methods=['GET', 'POST'])
def render_creditscoring_custom():
    form_content = csmodel.get_form_content()
    return render_template(
        "creditscoring_input.html",
        forms=form_content
    )

@application.route("/creditscoring_results", methods=['GET', 'POST'])
def render_creditscoring_results():
    if request.method == 'POST':
        inputs = {}
        table_inputs = {}
        for i, fn in enumerate(csmodel.feature_names):
            if i in csmodel.categorical_feature_indices:
                custom_input = int(request.form.get(fn))
                table_inputs[fn] = csmodel.encoders[i].classes_[custom_input]
                inputs[fn] = custom_input
            else:
                candidates = list(
                                enumerate(
                                    sorted(csmodel.raw[fn].unique().tolist()
                                    )
                                )
                             )
                custom_input = int(request.form.get(fn))
                inputs[fn] = candidates[custom_input][1]
                table_inputs[fn] = candidates[custom_input][1]
        custom_df = pd.DataFrame.from_dict(inputs, orient="index").T
        custom_table_df = pd.DataFrame.from_dict(table_inputs, orient="index").T
        custom_exp = csmodel.get_custom_explanation(custom_df.iloc[0].as_matrix())
        custom_table = custom_table_df.to_html(classes="table table-striped .table-condensed")
        return render_template(
            "creditscoring_results.html",
            custom_table=custom_table,
            custom_exp=custom_exp
        )

@application.route("/petimages")
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


@application.route("/textdetox")
def render_textdetox():
    logger.info("Creating random text examples.")
    randoms = [np.random.randint(0, len(tmodel.toxic_labels))
               for i in range(3)]
    random_exps = []
    for r in randoms:
        idx = tmodel.non_toxic_labels[r]
        random_exps.append(tmodel.get_explanation(idx))
        idx = tmodel.toxic_labels[r]
        random_exps.append(tmodel.get_explanation(idx))
    return render_template(
        'textdetox.html',
        random_exps=random_exps
    )


if __name__ == '__main__':
    application.run(host="0.0.0.0", debug=False)
