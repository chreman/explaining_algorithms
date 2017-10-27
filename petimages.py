import os
import numpy as np
import keras
from keras.applications import inception_v3 as inc_net
from keras.applications import resnet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import mpld3


def transform_img_fn(path_list):
    out = []
    valid_files = []
    for i, img_path in enumerate(path_list):
        # img = image.load_img(img_path, target_size=(299, 299))
        img = misc.imread(img_path)
        if len(img.shape) != 3:
            continue
        if len(img.shape) == 3:
            imx, imy, _ = img.shape
            if imx < 500:
                img = misc.imresize(img, 500/imx)
            imx, imy, _ = img.shape
            if imy < 500:
                img = misc.imresize(img, 500/imy)
            img = img[0:500, 0:500]
            img = misc.imresize(img, 300/500)
            img = img[0:299, 0:299]
        if img.shape != (299, 299, 3):
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
        valid_files.append(i)
    return np.vstack(out), valid_files


class NNModel(object):
    """docstring for NNModel"""
    def __init__(self, model, logger,
                 datapath="/home/chris/data/ML/images/oxfordiiipets"):
        super(NNModel, self).__init__()
        self.datapath = datapath
        self.model = model
        self.logger = logger

    def load_raw_data(self):
        self.image_files = os.listdir(os.path.join(self.datapath,
                                                   "images"))
        # self.trimap_files = os.listdir(os.path.join(self.datapath,
        #                                             "annotations", "trimaps"))
        # self.xml_files = os.listdir(os.path.join(self.datapath,
        #                                          "annotations", "xmls"))

    def preprocess(self):
        self.load_raw_data()
        self.images, valid_files = transform_img_fn(
                                    [os.path.join(self.datapath,
                                                  "images",
                                                  i)
                                     for i in self.image_files[:100]])
        self.valid = [self.image_files[i] for i in valid_files]

    def create_model_explainer(self):
        self.preds = self.model.predict(self.images)
        self.decoded_preds = decode_predictions(self.preds, )
        self.explainer = LimeImageExplainer()

    def get_explanation(self, i):
        title_text = self.valid[i].split("_")[:-1]
        title_text = " ".join(title_text)
        num_labels = 3
        preds = self.decoded_preds[i]
        explanation = self.explainer.explain_instance(
                            self.images[i],
                            self.model.predict,
                            top_labels=num_labels,
                            hide_color=0, num_samples=200)
        fig, subplots = plt.subplots(1, num_labels)
        fig.set_figheight(3)
        fig.set_figwidth(9)
        j = 0
        for exp, ax in zip(explanation.local_exp.keys(), subplots):
            temp, mask = explanation.get_image_and_mask(
                            exp,
                            positive_only=False,
                            num_features=5,
                            hide_rest=False)
            ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            ax.set_title(preds[j][1] + ", %.3f" % preds[j][2])
            j += 1
        plt.tight_layout()
        return mpld3.fig_to_html(fig), title_text
