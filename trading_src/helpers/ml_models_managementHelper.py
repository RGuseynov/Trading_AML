import os
import glob
import json
import pickle

from pathlib import Path


class ManagementHelper():

    # creation of folders structure at initialisation of an instance of the class
    def __init__(self, ml_experiment:dict):
        self.ml_experiment = ml_experiment
        # creation of model name folder if not exist
        self.model_name_folder = "ml_models/" + self.ml_experiment["general_informations"]["model_name_folder"]
        Path(self.model_name_folder).mkdir(parents=True, exist_ok=True)
        # creation of iterations folder in model folder
        Path(self.model_name_folder + "/iterations_uniques").mkdir(parents=True, exist_ok=True)
        Path(self.model_name_folder + "/iterations_sets").mkdir(parents=True, exist_ok=True)
        if self.ml_experiment["general_informations"]["iteration_set"]:
            # retrieving iteration set number and saving in experiment and in iteration_set attribut for easy acces
            self.ml_experiment["general_informations"]["iteration_set"] = self._get_count_iterations("iteration_set") 
            self.iteration_set = self.ml_experiment["general_informations"]["iteration_set"]
            self.iteration_set_folder_path = self.model_name_folder + "/iterations_sets/iteration_set_" + str(self.iteration_set)
            # creation of the specific iteration_set folder, models subfolder and jsons subfolder 
            # all models of the iteration_set will be in the same subfolder with
            # all corresponding jsons of the iteration_set in an other subfolder
            Path(self.iteration_set_folder_path).mkdir()
            Path(self.iteration_set_folder_path + "/models_files_" + str(self.iteration_set)).mkdir()
            Path(self.iteration_set_folder_path + "/json_files_" + str(self.iteration_set)).mkdir()
        else:
            self.ml_experiment["general_informations"]["iteration_unique"] = self._get_count_iterations("iteration_unique")
            # creation of the specific iteration_unique folder
            self.iteration_unique_folder_path = self.model_name_folder + "/iterations_uniques/iteration_unique_" + str(self.ml_experiment["general_informations"]["iteration_unique"])
            Path(self.iteration_unique_folder_path).mkdir()

    
    # update iteration_count file wich track the number of iterations_uniques and iterations_sets
    # and return iteration_count corresponding value
    def _get_count_iterations(self, unique_or_set:"string") -> int:
        if Path(self.model_name_folder + '/count_iterations.json').is_file():
            count_iterations = {}
            with open(self.model_name_folder + '/count_iterations.json', 'r+') as file:
                count_iterations = json.load(file)
                count_iterations[unique_or_set] += 1
                file.seek(0) 
                json.dump(count_iterations, file, indent=4, default=str)
            return count_iterations[unique_or_set]
        else:
            count_iterations = {"iteration_unique": 0, "iteration_set": 0}
            count_iterations[unique_or_set] += 1
            with open(self.model_name_folder + '/count_iterations.json', 'w') as file:
                json.dump(count_iterations, file, indent=4, default=str)
            return count_iterations[unique_or_set]


    def _reorder_ml_experiment_dict(self):
        #k = {k : Not_Ordered[k] for k in key_order}
        #key_order = ["model_name_folder", "iteration_set",]
        #self.ml_experiment = {key : self.ml_experiment[key] for key in key_order}
        pass


    def save_experiment(self, model, neural_network=False):
        #self._reorder_ml_experiment_dict()
        if self.ml_experiment["general_informations"]["iteration_set"]:
            with open(self.iteration_set_folder_path + "/json_files_" + str(self.iteration_set) +
                      "/training_info_" + str(self.ml_experiment["general_informations"]["iteration"]) + ".json", 'w') as file:
                json.dump(self.ml_experiment, file, indent=4, default=str)
            #model saving
            if neural_network:
                model.save(self.iteration_set_folder_path + "/models_files_" + 
                           str(self.iteration_set) + "/model_" + str(self.ml_experiment["general_informations"]["iteration"]))
            else:
                pickle.dump(model, open(self.iteration_set_folder_path + "/models_files_" + 
                                        str(self.iteration_set) + "/model_" + str(self.ml_experiment["general_informations"]["iteration"]) + ".pkl", 'wb'))
        else: 
            with open(self.iteration_unique_folder_path + "/training_info.json", 'w') as file:
                json.dump(self.ml_experiment, file, indent=4, default=str)
            #model saving
            if neural_network:
                model.save(self.iteration_unique_folder_path + "/model")
            else:
                pickle.dump(model, open(self.iteration_unique_folder_path + "/model.pkl", 'wb'))

    
    def save_history(self, history):
        if self.ml_experiment["general_informations"]["iteration_set"]:
            pass
        else:
            pass


    # saving results of iteration set by oredering from best to lowest
    def save_iteration_results(self, results):
        results_ordered = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        with open(self.iteration_set_folder_path + "/results_ordered.json", 'w') as file:
            json.dump(results_ordered, file, indent=4, default=str)

