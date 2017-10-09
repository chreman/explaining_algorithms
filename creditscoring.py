import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from lime.lime_tabular import LimeTabularExplainer


class CSModel(object):
    """docstring for CSModel"""
    def __init__(self, datapath, logger):
        super(CSModel, self).__init__()
        self.datapath = datapath
        self.logger = logger

    def load_raw_data(self):
        with open(os.path.join(self.datapath,
                               "german_credit_preprocessed.csv"),
                  "r") as infile:
            raw = pd.read_csv(infile)
        return raw

    def preprocess(self):
        self.raw = self.load_raw_data()
        self.labels = self.raw['Creditability']
        self.features = self.raw[self.raw.columns[1:]]
        self.feature_names = self.features.columns
        self.class_names = ['Not Credit Worthy', 'Credit Worthy'] # [0, 1]
        self.categorical_features = ['Account Balance',
                                     'Payment Status of Previous Credit',
                                     'Purpose', 'Value Savings/Stocks',
                                     'Length of current employment',
                                     'Sex & Marital Status',
                                     'Most valuable available asset',
                                     'Concurrent Credits', 'Type of apartment',
                                     'Occupation', 'Telephone',
                                     'Foreign Worker', 'Guarantors']
        self.categorical_feature_indices = [self.features.columns.tolist().index(cf)
                                            for cf in self.categorical_features]

    def create_model(self):
        features = self.features.as_matrix()
        self.categorical_names = {}
        for feature in self.categorical_feature_indices:
            le = LabelEncoder()
            le.fit(features[:, feature])
            features[:, feature] = le.transform(features[:, feature])
            self.categorical_names[feature] = le.classes_
        self.ohe = OneHotEncoder(categorical_features=self.categorical_features)
        train, test, labels_train, labels_test = train_test_split(features,
                                                                  self.labels,
                                                                  train_size=0.80)
        self.train = train
        self.test = test
        self.labels_train = labels_train.as_matrix().reshape(-1, 1)
        self.labels_test = labels_test.as_matrix().reshape(-1, 1)
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(self.train, self.labels_train.ravel())

    def create_model_explainer(self):
        self.explainer = LimeTabularExplainer(
                          self.train,
                          feature_names=self.feature_names,
                          training_labels=self.labels_train,
                          class_names=self.class_names,
                          categorical_features=self.categorical_feature_indices,
                          categorical_names=self.categorical_names,
                          discretize_continuous=True
                         )

    def get_explanation(self, i):
        exp = self.explainer.explain_instance(self.test[i],
                                              self.classifier.predict_proba,
                                              num_features=4, top_labels=1)
        return (exp.as_html(show_table=True, show_all=False,
                            show_predicted_value=True, predict_proba=True),
                self.labels_test[i])

    def remap_categoricals(self, df):
        for feature, categories in self.categorical_names.items():
            featurename = self.feature_names[feature]
            df[featurename] = df[featurename].map(lambda x: list(categories)[x])
        return df


def main():
    pass


if __name__ == '__main__':
    main()
