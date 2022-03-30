import utilities
import pandas as pd
import numpy as np
import pysd
from datetime import date

# Load the model.
model = pysd.load("model.py")

# Set the start and end dates. 
start_date =  date(year=2001, month=1, day=1)
end_date =  date(year=2021, month=12, day=1)

# Set the lockdown dates.
lockdown_start_date = date(year=2020, month=3, day=23)
lockdown_end_date = date(year=2020, month=6, day=15)

# Set the number of pre lockdown days to use for the optimisation procedure.
no_pre_lockdown_days = 360

# Set the number of days for the projection period.
no_projection_days = 360

# Calculate the number of lockdown days. 
no_lockdown_days = (lockdown_end_date - lockdown_start_date).days

# Set post lockdown start and end date.
post_lockdown_start_date = lockdown_end_date
post_lockdown_end_date = date(year=2021, month=12, day=1)

# Calculate the number of post lockdown days.
no_post_lockdown_days = (post_lockdown_end_date - post_lockdown_start_date).days

# Calculate the total number of simulation days.
no_simulation_days = no_pre_lockdown_days + no_lockdown_days\
                    + no_post_lockdown_days + no_projection_days

# Store the dates in a dictionary.
dates = {
            "start date": start_date,
            "end date": end_date,
            "lockdown start date": lockdown_start_date,
            "lockdown end date": lockdown_end_date,            
}

# Generate the time periods.
time_periods, dates = utilities.generate_time_periods(
                                            dates, 
                                            no_pre_lockdown_days,
                                            no_post_lockdown_days,
                                            no_projection_days
)

time_period_labels = list(time_periods.keys())


# Load dictionary 
data_dictionary = np.load("data_dictionary.npy", allow_pickle=True).flat[0]

# Set the data plotting parameters
data_plotting_parameters = {
                                "data dictionary": data_dictionary,
                                "dates": dates, 
                                "show flag": False,
                                "save flag": True,
                                "text fontsize": 30,
                                "ticks fontsize": 25,
                                "data type": "raw"
}

# Plot data if you want to.
#utilities.plot_all_data(data_plotting_parameters)

# Set optimisation parameters.
alpha = 0.001
beta = utilities.compute_beta(data_dictionary)
delta_1 = 1e-4
delta_2 = 1e-2
time_step = 1
time_step_factor = int(1/time_step)
no_time_steps = no_simulation_days*time_step_factor
no_simulation_runs = 100
max_iters = 1000


no_days_dictionary = {
                        "pre lockdown": no_pre_lockdown_days,
                        "lockdown": no_lockdown_days,
                        "post lockdown": no_post_lockdown_days,
                        "projection": no_projection_days           
}

# Optimise model parameters.
utilities.optimise_model_parameters(
                    dates,
                    data_dictionary,
                    time_periods,
                    alpha,
                    beta,
                    delta_1,
                    delta_2,
                    no_days_dictionary,
                    time_step,
                    model,
                    max_iters
)

quit()

stock_names = [
                "Symptomatic population",
                "GP waiting list population",
                "Diagnostic waiting list population",
                "Consultant waiting list population",
                "Treatment waiting list population"
]

intervention_dictionary = {   
                    "beta_M": {"flag": False, "multiplier": None},
                    "beta_S": {"flag": False, "multiplier": None},
                    "B_GPL": {"flag": False, "multiplier": None},
                    "alpha_G": {"flag": False, "multipliers": None},
                    "alpha_D": {"flag": False, "multiplier": None},
                    "alpha_C": {"flag": False, "multiplier": None},
                    "alpha_T": {"flag": False, "multiplier": None},
                    "beta_SG": {"flag": False, "multiplier": None},                
                    "beta_GC": {"flag": False, "multiplier": None},
                    "beta_GD": {"flag": False, "multiplier": None},
                    "beta_GS": {"flag": False, "multiplier": None},
                    "beta_DC": {"flag": False, "multiplier": None},
                    "beta_DG": {"flag": False, "multiplier": None}, 
                    "beta_CT": {"flag": False, "multiplier": None},
                    "beta_CD": {"flag": False, "multiplier": None}, 
                    "beta_CC": {"flag": False, "multiplier": None},
                    "beta_CG": {"flag": False, "multiplier": None},
}

figure_labels = {
                    "all": [0.1, 0.05, 0.0],
                    "alpha_T": [0.1, 0.05, 0.0],            
                    "alpha_C": [0.1, 0.05, 0.0],
                    "alpha_G": [0.1, 0.05, 0.0],
                    "alpha_D": [0.1, 0.05, 0.0],
}



for figure_label, values in figure_labels.items():
    
    model_outputs_dict = {}
    
    for value in values:
        
        
        if figure_label!="all":
           
            intervention_dictionary[figure_label]["flag"] = True
            intervention_dictionary[figure_label]["multiplier"] = 1 + value
        
            print("------------------------------------------------")
            print("Multiplier: ", intervention_dictionary[figure_label]["multiplier"])
            figure_label = figure_label + "_" + str(int(value*100))
            print("Figure label: ", figure_label)

        else:
            
            intervention_dictionary["alpha_G"]["flag"] = True
            intervention_dictionary["alpha_G"]["multiplier"] = 1 + value
            
            intervention_dictionary["alpha_D"]["flag"] = True
            intervention_dictionary["alpha_D"]["multiplier"] = 1 + value
            
            intervention_dictionary["alpha_C"]["flag"] = True
            intervention_dictionary["alpha_C"]["multiplier"] = 1 + value
            
            intervention_dictionary["alpha_T"]["flag"] = True
            intervention_dictionary["alpha_T"]["multiplier"] = 1 + value
            
            print("Figure label: ", figure_label)

        print("------------------------------------------------")
        simulation_parameters = {
                                    "no simulation runs": no_simulation_runs,
                                    "no time steps": no_time_steps,
                                    "time step": time_step,
                                    "no simulation days": no_simulation_days,
                                    "model": model,
                                    "no days dictionary": no_days_dictionary,
                                    "intervention dictionary": intervention_dictionary,
        }

        model_outputs = utilities.run_monte_carlo_simulation(simulation_parameters)

        model_outputs_dict[str(int(value*100))] = utilities.extract_output_statistics(model_outputs)

        if figure_label!="all":
                   
            figure_label = figure_label.replace("_" + str(int(value*100)), "")
            intervention_dictionary[figure_label]["flag"] = False
            intervention_dictionary[figure_label]["multiplier"] = 1 
            
        else:
            intervention_dictionary["alpha_G"]["flag"] = False
            intervention_dictionary["alpha_G"]["multiplier"] = 1 
            
            intervention_dictionary["alpha_D"]["flag"] = False
            intervention_dictionary["alpha_D"]["multiplier"] = 1 
            
            intervention_dictionary["alpha_C"]["flag"] = False
            intervention_dictionary["alpha_C"]["multiplier"] = 1 
            
            intervention_dictionary["alpha_T"]["flag"] = False
            intervention_dictionary["alpha_T"]["multiplier"] = 1 
            
    model_plotting_parameters = {
                                    "data": data_dictionary,
                                    "model outputs": model_outputs_dict, 
                                    "stock names": stock_names,                            
                                    "dates": dates, 
                                    "show flag": False,
                                    "save flag": True,
                                    "text fontsize": 30,
                                    "ticks fontsize": 25,
                                    "figure label": figure_label
    }

    utilities.plot_all_model_outputs(model_plotting_parameters)

