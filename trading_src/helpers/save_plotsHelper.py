import pandas as pd

import xgboost as xgb
from xgboost import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import graphviz
import seaborn as sns

import pickle


class XGBoostPlotHelper():

    def __init__(self, model, destination_path=None):
        self.model = model
        self.destination_path = destination_path

    def show_importance(self):
        fig, ax = plt.subplots(figsize=(10,10))
        xgb.plot_importance(self.model, ax=ax, max_num_features=max_num_features)
        plt.show()

    def save_importance(self, max_num_features=25):
        fig, ax = plt.subplots(figsize=(10,10))
        xgb.plot_importance(self.model, ax=ax, max_num_features=max_num_features)
        plt.savefig(self.destination_path + "/features_importance.png")

    def get_features_importances(self, max_num_features=25):
        fig, ax = plt.subplots(figsize=(20,20))
        xgb.plot_importance(self.model, ax=ax, max_num_features=max_num_features)
        return fig

    def show_tree(self):
        fig, ax = plt.subplots(figsize=(28, 28))
        xgb.plot_tree(self.model, ax=ax, rankdir='LR', num_trees=num_trees)
        plt.show()

    def save_tree(self, num_trees):
        fig, ax = plt.subplots(figsize=(28, 28))
        xgb.plot_tree(self.model, ax=ax, rankdir='LR', num_trees=num_trees)
        plt.savefig(self.destination_path + "/tree_num_" + str(num_trees) + ".png")

    def show_binary_confusion_matrix(self):
        cm_df = pd.DataFrame(confusion_matrix,
                     index = ['down', 'up'],
                     columns = ['down','up'])
        plt.figure(figsize=(5.5,4))
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def save_binary_confusion_matrix(self, confusion_matrix):
        cm_df = pd.DataFrame(confusion_matrix,
                     index = ['down', 'up'],
                     columns = ['down','up'])
        plt.figure(figsize=(5.5,4))
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.destination_path + "/confusion_matrix.png")


if __name__ == "__main__":
    #bst = xgb.Booster()
    #bst.load_model('csv1_XGBOOST_BINLOG_1MIN_TO_5MIN_Later_.model') 
    xgb_model = pickle.load(open("Models/ALL_XGBOOST_BINLOG_1MIN_TO_5MIN_Later.pkl", 'rb'))

    print(xgb_model.feature_names)

    xgb.plot_importance(xgb_model, max_num_features=25)

    plt.show()

    #xgb.plot_tree(xgb_model, num_trees=2)
    #plt.show()

    fig, ax = plt.subplots(figsize=(28, 28))
    xgb.plot_tree(xgb_model, ax=ax, rankdir='LR', num_trees=9)
    plt.savefig("temp.pdf")
    plt.show()
