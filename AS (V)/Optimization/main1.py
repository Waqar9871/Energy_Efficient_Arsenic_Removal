import pandas as pd
import numpy as np
import pickle
from geneticalgorithm import geneticalgorithm as ga

# Load the pre-trained model and data
model_cat = pickle.load(open('model.sav', 'rb'))
df = pd.read_csv('data.csv')

# Define the objective function
def objective(params):
  prediction=model_cat.predict(params)
  prediction=(prediction-df.target.min())/(df.target.max()-df.target.min())
  rt=(params[-5]-df.rt.min())/(df.rt.max()-df.rt.min())
  e=(params[-2]-df.e.min())/(df.e.max()-df.e.min())
  score = -prediction +  rt + e
  return score

# Define the genetic algorithm function
def gas(varbound):
    """
    Run genetic algorithm optimization.
    
    Parameters:
    varbound (ndarray): Array of variable boundaries for optimization.
    
    Returns:
    tuple: The final score, model prediction, and genetic algorithm report.
    """
    ga_model = ga(
        function=objective,
        dimension=11,
        variable_type_mixed=np.array(['int', 'int', 'int', 'real', 'real', 'real', 'real', 'real', 'real', 'real', 'int']),
        variable_boundaries=varbound,
        algorithm_parameters={
            'max_num_iteration': 500,
            'population_size': 200,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }
    )
    ga_model.run()
    final_params = ga_model.output_dict['variable']
    final_score = -model_cat.predict(final_params) + final_params[-5] + final_params[-2]
    print(f"Final Score: {final_score}, Model Prediction: {model_cat.predict(final_params)}")
    return ga_model.output_dict['variable'],final_score, model_cat.predict(final_params), ga_model.report

# Define the optimization function
def run_optimization(t):
    """
    Run optimization for a given t value.
    
    Parameters:
    t (float): Fixed value for 'asc' parameter.
    """
    varbound = np.array([
        [df.mt.min(), df.mt.max()],
        [df.btu.min(), df.btu.max()],
        [df.bt.min(), df.bt.max()],
        [df.bet.min(), df.bet.max()],
        [df.p.min(), df.p.max()],
        [df.ph.min(), df.ph.max()],
        [df.rt.min(), df.rt.max()],
        [t, t],  # 'asc' parameter fixed
        [df.ad.min(), df.ad.max()],
        [df.e.min(), df.e.max()],
        [df.pt.min(), df.pt.max()]
    ])
    
    results = {}
    for n in range(10):
        print(f"t={t}, iteration={n}", end=',')
        results[n] = gas(varbound)
        
    results_df = pd.DataFrame(results, index=['input_parameters','objective_function', 'model_prediction', 'result'])
    results_df.to_csv(f"optimizing_results_{t}.csv")

# Example usage: run optimization for the first t value
t_values = [0.6710000000000002, 2.2860000000000005, 4.993, 9.97, 10.8, 20.0, 40.0, 74.97000000000001, 100.0, 1001.07]
run_optimization(t_values[1])
