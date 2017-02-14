from .base import InvalidInput, hist_to_title, adv_getitem
import numpy as np
import pandas as pd
import copy
import warnings

from sklearn import svm

import logging
log=logging.getLogger(__name__)
class LearnMixin():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_commands.update({"SVM": self.support_vector_machine, "PREDICT":self.predict})
        self._trained_classifiers = {}
    def support_vector_machine(self, svm_name, _from, key, _on=None, *selected_keys):
        """
        
        Use `SVM svm_name FROM key`
        """
        if _from != "FROM":
            raise InvalidInput("Wrong syntax for SVM. Expected 'FROM', found '{}'".format(as_))
        if key not in self.filtered_data:
            raise InvalidInput("Key '{}' not present".format(key))
        if self.filtered_data[key].dtype == np.float_:
            clf = svm.SVR()
        else:
            clf = svm.SVC()
        X = self.filtered_data.drop(key, axis=1, inplace=False)
        if _on is not None:
            if _on != "ON":
                raise InvalidInput("Expected 'ON', found '{}'".format(as_))
            try:
                X = X[list(selected_keys)]
            except KeyError as e:
                raise InvalidInput("Invalid key in {}: {}".format(selected_keys, e)) from e
        X = X.select_dtypes(include=[np.number])
        headers = X.columns.values
        print(headers)
        clf.fit(X, self.filtered_data[key])
        clf._fav_headers = headers
        clf._fav_trainedon_history = copy.copy(self.filtered_data._fav_history)
        clf._fav_trainedon_datasetname = self.filtered_data._fav_datasetname
        clf._fav_target_column = key

        self._trained_classifiers[svm_name]=clf
        print("Support Vector Machine was trained and stored as {}".format(svm_name))
        print("Use 'PREDICT {}' to use it.".format(svm_name))
    def predict(self, clf_name, _as = None, column_name = None):
        """
        Use a classifier that has been trained before, to do predictions on the current dataset.
        
        Use 'PREDICT classifier_name AS column_name'. 
        The predictions are stored in the column `column_name`, which is then added to the dataframe.
        """
        clf = self._trained_classifiers[clf_name]           
        key = clf._fav_target_column
        headers = clf._fav_headers
        try:
            X = self.filtered_data[headers]
        except Exception as e:
            raise InvalidInput("Cannot use this classifier on this data, because it does not "
                               "contain the same columns as the trainings data.") from e
        y = clf.predict(X)
        if _as is None:
            print(y)
        elif _as != "AS":
            raise InvalidInput("Expecting 'AS', found '{}'".format(y))
        else:
            if column_name is None:
                raise InvalidInput("Please provide a new columnname after 'AS'")            
            
            #Change the base dataset to add the column. Then recalculate all stored "views" based on their stored history.

            data = self.data[self.filtered_data._fav_datasetname]
            if column_name in data.columns.values:
                warnings.warn("Overwriting column {}".format(column_name))
            else:
                data[column_name] = np.nan
                log.info("Column {} created for dataset {}".format(column_name, self.filtered_data._fav_datasetname))
            for i,row in enumerate(self.filtered_data.index):
                assert (data[headers].ix[row]==np.asarray(X)[i]).all(), "{}:\n{}, {}".format(data[headers].ix[row], np.array(X)[i], X.ix[row])
                data.loc[row,column_name]=y[i]
            for key, stored_dataset in self.stored.items():
                if stored_dataset._fav_datasetname != data._fav_datasetname:
                    continue
                self.stored[key] = self.replay_history(data, stored_dataset._fav_history)
            self.filtered_data = self.replay_history(data, self.filtered_data._fav_history)