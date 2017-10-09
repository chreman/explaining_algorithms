import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


class TextModel(object):

    def __init__(self, logger,
                 datapath="/home/chris/data/WMTalk/toxicity/",
                 modelpath="data/tmodel.pkl"):
        super(TextModel, self).__init__()
        self.datapath = datapath
        self.modelpath = modelpath
        self.stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                      'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                      'about', 'against', 'between', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                      'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                      'can', 'will', 'just', 'don', 'should', 'now']
        self.class_names = ['non_toxic', 'toxic']

    def load_raw_data(self):
        self.toxicity_annotations = pd.read_csv(os.path.join(
                                        self.datapath,
                                        "toxicity_annotations.tsv"), sep="\t")
        self.toxicity_annotated_comments = pd.read_csv(os.path.join(
                                        self.datapath,
                                        "toxicity_annotated_comments.tsv"),
                                        sep="\t")


    def preprocess(self):
        self.load_raw_data()
        median_toxicity = self.toxicity_annotations.groupby('rev_id')['toxicity'].agg('median')
        self.toxicity_annotated_comments['comment'] = self.toxicity_annotated_comments['comment'].map(lambda x: x.replace("NEWLINE_TOKEN", " ").lower())
        features = self.toxicity_annotated_comments['comment'].map(lambda x: x.split())
        features = features.map(lambda x: [token for token in x if token not in self.stops])
        self.features = features.map(lambda x: " ".join(x))
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.features)
        ss = ShuffleSplit(n_splits=1, test_size=0.3)
        self.train_index, self.test_index = list(ss.split(self.vectors))[0]
        self.labels = median_toxicity.as_matrix().astype(int)
        self.toxic_labels = [i for i, l in enumerate(self.labels) if l == 1]
        self.non_toxic_labels = [i for i, l in enumerate(self.labels) if l == 0]

    def create_model(self):
        with open(self.modelpath, "rb") as infile:
            classifier = pickle.load(infile)
        self.pipeline = make_pipeline(self.vectorizer, classifier)

    def create_model_explainer(self):
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def get_explanation(self, i):
        exp = self.explainer.explain_instance(
                        self.toxicity_annotated_comments['comment'][i],
                        self.pipeline.predict_proba, num_features=5)
        return exp.as_html()
