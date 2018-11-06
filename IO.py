import pandas as pd 
import numpy as np 

class ProteinIO:
  @staticmethod
  def get_input(filename):
    return pd.read_csv(filename)
  
  @staticmethod
  def get_output(filename):
    output = []
    file = np.load(filename)
    for key in file:
      output.append(file[key])
    return np.array(output)

  @staticmethod
  def save_predictions(predictions):
    np.savez('test_{}.npz'.format(5), *predictions)
  
  @staticmethod
  def calculate_rmse(solutions_filename, predictions_filename):
    solution_file = np.load(solutions_filename)
    solutions = []
    for key in solution_file:
      solutions.append(solution_file[key])
    
    predictions_file = np.load(predictions_filename)
    predictions = []
    for key in predictions_file:
      predictions.append(predictions_file[key])
    
    return np.sum((solutions - predictions) ** 2)
    



    
  
  