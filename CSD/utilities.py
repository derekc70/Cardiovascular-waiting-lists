"""

This module contains utility functions for all other modules. 

"""

## Imports 
import copy
from re import L
import selenium
import os
import datetime
import pandas as pd
import numpy as np 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import Request, urlopen
import datetime
from datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt 
from matplotlib import figure
from scipy import optimize as opt
from scipy import interpolate
from sklearn.metrics import mean_squared_error 
from scipy.optimize import Bounds, LinearConstraint, minimize


plt.style.use('ggplot')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1

# Hard coded lists and dictionaries that are used in some of the functions.
annual_width = 300
monthly_width = 20

data_plotting_names = {
                        "p_conds": ["Cumulative prevalence rate", "Annual rate (%)", annual_width],
                        "p_conds_smoothed": ["Cumulative prevalence rate", "Annual rate (%)", annual_width],
                        "N_L_E": ["People registered at a GP practice in England", "People", annual_width],
                        "N_L_W": ["People registered at a GP practice in Wales", "People", annual_width],
                        "N_IC": ["Cumulative deaths with cardiology ICD-10 codes", "People", annual_width],
                        "W_MRI_cardiology": ["Waiting list for cardiology MRI", "People", monthly_width],
                        "W_CT_cardiology": ["Waiting list for cardiology CT", "People", monthly_width],
                        "W_echocardiography": ["Waiting list for echocardiography", "People", monthly_width],
                        "W_electrophysiology": ["Waiting list for electrophysiology", "People", monthly_width],
                        "N_MRI_cardiology": ["cardiology MRI tests", "Tests", monthly_width],
                        "N_CT_cardiology": ["cardiology CT tests", "Tests", monthly_width],
                        "N_echocardiography":  ["Echocardiography tests", "Tests", monthly_width],
                        "N_electrophysiology": ["Electrophysiology tests", "Tests", monthly_width],
                        "W_incomplete_cs": ["Incomplete RTT pathways for cardiothoracic surgery", "RTT pathways", monthly_width],
                        "W_incomplete_c": ["Incomplete RTT pathways for cardiology", "RTT pathways", monthly_width],
                        "W_incomplete_dta_cs": ["Incomplete RTT pathways for cardiothoracic surgery with DTA", "RTT pathways", monthly_width],
                        "W_incomplete_dta_c":  ["Incomplete RTT pathways for cardiology with DTA", "RTT pathways", monthly_width],
                        "N_completed_admitted_cs": ["Completed cardiothoracic surgery RTT pathways with admission", "RTT pathways", monthly_width],
                        "N_completed_admitted_c":["Completed cardiology RTT pathways with admission", "RTT pathways", monthly_width],
                        "N_completed_nonadmitted_cs": ["Completed cardiothoracic surgery RTT pathways without admission", "RTT pathways", monthly_width],
                        "N_completed_nonadmitted_c": ["Completed cardiology RTT pathways without admission", "RTT pathways", monthly_width],
                        "N_new_cs": ["New cardiothoracic surgery RTT pathways", "RTT pathways", monthly_width],
                        "N_new_c": ["New cardiology RTT pathways", "RTT pathways", monthly_width],
                        "A_GP": ["GP appointments", "Appointments", monthly_width],
                        "B_GP": ["GP appointment bookings", "Bookings", monthly_width],
                        "N_FCE_codes": ["Finished consultant episodes for cardiology codes", "FCE", monthly_width],
                        "N_FCEP_codes": ["Finished consultant episodes for cardiology codes with a procedure", "FCEP", monthly_width],
                        "N_A_codes": ["Consultant appointments for cardiology codes", "Appointments", monthly_width],
                        "absence_rate": ["Staff absence rate", "Monthly rate (%)", monthly_width],
}


stocks_plotting_names = {
                            "P_S": "Symptomatic population",
                            "P_G": "GP waiting list population",
                            "P_D": "Diagnostic waiting list population",
                            "P_C": "Consultant waiting list population",
                            "P_T": "Treatment waiting list population"
}

parameters_plotting_names = {
                                "beta_M": ["Probability of dying with symptoms", "Probability"],
                                "prob_neg_M": ["Probability of not dying with symptoms", "Probability"],
                                "alpha_G": ["GP appointment supply", "Appointments"],
                                "B_GPL": ["GP appointment bookings", "Appointments"],
                                "alpha_C": ["Consultant appointment supply", "Appointments"],
                                "alpha_D": ["Diagnostic appointment supply","Appointments"],
                                "alpha_T": ["Treatment appointment supply","Appointments"],
                                "beta_S": ["Symptom development rate", "Patients"],
                                "beta_GS": ["Proportion of GP appointments with discharge to symptomatic population",
                                                        "Proportion"],
                                "beta_GC": ["Proportion of GP appointments with consultant referral",
                                                        "Proportion"],
                                "beta_GD": ["Proportion of GP appointments with diagnostic referral",
                                                         "Proportion"],
                                "beta_DC": ["Proportion of diagnostic appointments with consultant referral",
                                                        "Proportion"],
                                "beta_DG": ["Proportion of diagnostic appointments with discharge to GP", 
                                                        "Proportion"],
                                "beta_CT": ["Proportion of consultant appointments with treatment referral",
                                                        "Proportion"],
                                "beta_CD": ["Proportion of consultant appointments with diagnostic referral", 
                                                        "Proportion"],                            
                                "beta_CG": ["Proportion of consultant appointments with discharge to GP",
                                                        "Proportion"],                                
                                "beta_SG": ["Proportion of symptomatic patients booking GP appointments",
                                                        "Proportion"],                                
                                "beta_CC": ["Proportion of consultant appointments with consultant referral",
                                                        "Proportion"],

    }

conditions = ["AF", "CHD", "HF", "HYP"]
ICD_10_cv = ["I" + str(x) for x in range(0, 530)] + ["I0" + str(x) for x in range(0, 100)]
codes = ["170", "172", "174", "320", "328", "331"]
months = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
months_cap = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
months_abr = ["jan", "feb", "mar", "apr", "may", "jun", "jul",
          "aug", "sep", "oct", "nov", "dec"]
months_hes = ["31JAN", "28FEB", "31MAR", "30APR", "31MAY", "30JUN",
              "31JUL", "31AUG", "30SEP", "31OCT", "30NOV", "31DEC"]
month_lengths = [31, 28, 31, 30, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_end_t_points = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
quarter_end_t_points = [365 * x for x in [1, 1/4, (1/2), (3/4)]]

"""======================================================================================================================"""
"""======================================================================================================================"""
### Data extraction functions 
"""======================================================================================================================"""
"""======================================================================================================================"""

def add_data_versions(data_dictionary: dict) -> dict:
    """ Creates two dictionary entries for each data source.
    
    This function creates a processed and a raw (key, value) pair
    in the data dictionary for each data source. 
    
    Parameters
    ----------
        data_dictionary: dict 
            Dictionary of data.
            
    Returns
    ----------
        data_dictionary: dict 
            Dictionary of data with processed and raw entries. 
    """
    # Loop through and add processed, raw (key, value) pairs.
    for data_name in list(data_dictionary.keys()):
        
        data_dictionary[data_name + "_processed"] = []
        data_dictionary[data_name + "_raw"] = []
        del data_dictionary[data_name]
        
    return data_dictionary


def reindex_data(
                    start_day_index: int,
                    end_day_index: int,
                    no_days: int,
                    data_dictionary: dict,
                    max_delay=False
    ):
    """

    This function deals with index discrepancies from the data
    extraction step. 
        
    Parameters
    ----------           
        start_day_index: int
            Index of first day data was available from that dataset.
        end_day_index: int
            Index of final day data was available from that dataset.
        no_days: int
            Total number of days to extract data for. 
        data_dictionary: dict 
            Dictionary of data with processed and raw entries. 
  
    Returns
    ----------
        data_dictionary: dict 
            Dictionary of data with processed and raw entries. 
    """  

    for data_name, data in data_dictionary.items():
        
        if not max_delay:
            
            data_dictionary[data_name] = resolve_index_discrepancy(
                                            start_day_index,
                                            end_day_index, 
                                            no_days,
                                            data
            )
            
        else:
            
            data_dictionary[data_name] = resolve_index_discrepancy(
                                            start_day_index,
                                            end_day_index - max_delay, 
                                            no_days,
                                            data
            )
             
    return data_dictionary        


def unique_list(non_unique_list: list) -> list:
    """ Converts non-unique list into a unique list.
    
    This function removes duplicate elements from a list. 
    
    Parameters
    ----------
        non_unique_list: list
            List with repeating elements.
            
    Returns
    ----------
        unique_list: list
            List without repeating elements.
    """
    unique_list = list(dict.fromkeys(non_unique_list))
    
    return unique_list


def initialise_selenium_browser():
    """ Initialises selenium browser for scraping links.
    
    This function initialises the selenium package with the Chrome 
    browser and a few options. 
        
    Returns
    ----------
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
    """
    # Set the options for selenium using the Chrome browser.
    chrome_options = webdriver.ChromeOptions() 

    # Prevent selenium from opening the browser each time. 
    chrome_options.add_argument("headless")

    # Open browser.
    browser = webdriver.Chrome(
                                ChromeDriverManager().install(),
                                options=chrome_options
    )
    
    return browser


def interpolate_data(
                        data: list, 
                        t_points: list, 
                        no_days: int,
                        interpolation_method: str
    ) -> np.array :
    """ Interpolates some data points.
    
    This function interpolates a list of data points. 
    
    Parameters
    ----------   
        data: list
            Time series of data to be interpolated.        
        t_points: list
            List of associated time points for the data. 
        no_days: int
            Number of days to interpolate the data for.
        interpolation_method: str
            String indicating type of interpolation to use.
    Returns
    ----------
        interpolated_data: array 
            Interpolated data.
    """
    # Our time unit is days, so convert to ints.
    t_points = np.floor(t_points).astype(np.int)

    if interpolation_method=="linear":
        
        # Set day range for interpolation.
        t_range = np.linspace(t_points[0], t_points[-1], no_days)
        
        # Perform linear interpolation. 
        interpolated_func = interpolate.interp1d(t_points, data, kind=interpolation_method)
            
        interpolated_data = interpolated_func(t_range)
    
    elif interpolation_method=="zeros":
        
        # This interpolation method just places the data point in the 
        # middle of the time period i.e. if it is monthly data then 
        # place the data point in the middle of the month.
        interpolated_data = []
    
        inter_data_periods = [t_points[0]-1] + [x - 1 for x in np.diff(t_points).tolist()]

        final_period = no_days - t_points[-1]
        
        for period, datum in zip(inter_data_periods, data):

            interpolated_data.extend([0]*period + [datum])
            
        interpolated_data.extend([0]*final_period)
        
        interpolated_data = np.asarray(interpolated_data)    
    
    return interpolated_data 


def interpolate_data_dictionary(
                                    data_dictionary: dict,
                                    t_points: list,
                                    no_interp_days: int,
                                    ignore_data_labels=[]
    ):    
    """ Interpolates a dictionary of time series data.
    
    This function interpolates a dictionary of time series data. 
    
    Parameters
    ----------   
        data_dictionary: dict
            Dictionary of time series to interpolate.  
        t_points: list
            List of associated time points for the data. 
        no_days: int
            Number of days to interpolate the series for.
        ignore_data_labels: list
            List of data labels indicating which series we don't want to interpolate.

    Returns
    ----------
        data_dictionary: dict 
            Dictionary of interpolated data.
    """
    # Loop through each time series in the dictionary and interpolate.
    for data_label, data in data_dictionary.items():
        
        # Ignore series if requested.
        if data_label in ignore_data_labels:
            pass
        
        else:     
                   
            if "raw" in data_label:
                interpolation_method = "zeros"
            
            elif "processed" in data_label:
                interpolation_method = "linear"

            data_dictionary[data_label] = interpolate_data(data, t_points, no_interp_days, interpolation_method)
        
    return data_dictionary


def extract_domain_links(
                            browser: selenium.webdriver.chrome.webdriver.WebDriver,
                            keywords: list,
                            file_extension: str
    ) -> list:
    """ Extracts the links in a given domain.
    
    This function extracts the links from a domain with a given file extension 
    using a list of keywords.
     
    Parameters
    ----------   
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        keywords: list 
            List of keywords to use to extract the correct links.
        file_extension: str
            File extension for the links, i.e. .xls or .csv.   
    
    Returns
    ----------
        dataset_links: list 
            Links to the desired datasets.
    """
    # Find the elements with the file extension as a substring.
    if file_extension == ".xls":
        elements = browser.find_elements_by_xpath('//a[contains(@href, ".xls")]')
    elif file_extension == ".csv":
        elements = browser.find_elements_by_xpath('//a[contains(@href, ".csv")]')
    elif file_extension == ".xlsx":
        elements = browser.find_elements_by_xpath('//a[contains(@href, ".xlsx")]')
    else:
        elements = browser.find_elements_by_xpath('//a[contains(@href, "")]')

    # Extract the those elements which are links.
    links = [element.get_attribute("href") for element in elements]
  
    # Count the number of keywords.
    no_keywords = len(keywords)

    # Initialise a list for the dataset links.
    dataset_links = []

    # Loop through the links in the domain and check for the keywords.
    for link in links:

        # If a link matches all the keywords then store it in the list.
        dataset_condition = sum([keyword in link for keyword in keywords]) == no_keywords

        if dataset_condition: 
            dataset_links.append(link)

    # Reverse the ordering so that the links are in the correct temporal order. 
    dataset_links.reverse()
    
    return dataset_links


def set_qof_indices(year: int) -> dict:
    """ Specify indices for qof data.

    This function sets the indices for the excel spreadsheets
    that contain the qof data for a given year period.
 
    Parameters
    ----------
        year: int
            The starting year of the period.

    Returns
    ----------
        index_dictionary: dict  
            Dictionary containing indices for the relavant data.
    """
    # Initialise dictionary.
    index_dictionary = {
                            "conditions": {},                                     
    }
    
    # Create dictionary for each year.
    if year==14:
        
        index_dictionary["conditions"] = {
                            "AF": {"row_N_L_E": 19, "col_N_L_E": 8, "row_p": 19, "col_p": 10},
                            "CHD":{"row_N_L_E": 19, "col_N_L_E": 8, "row_p": 19, "col_p": 10},
                            "HF":{"row_N_L_E": 19, "col_N_L_E": 8, "row_p": 19, "col_p": 10},
                            "HYP":{"row_N_L_E": 19, "col_N_L_E": 8, "row_p": 19, "col_p": 10},
        }

    elif year==15:
        
        index_dictionary["conditions"] = {
                            "AF": {"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
                            "CHD":{"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
                            "HF":{"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
                            "HYP":{"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
        }
        
    elif year in [16, 17]:
        
        index_dictionary["conditions"] = {
                            "AF": {"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
                            "CHD":{"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
                            "HF":{"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
                            "HYP":{"row_N_L_E": 9, "col_N_L_E": 8, "row_p": 9, "col_p": 10},
        }
        
    elif year == 18:
     
        
        index_dictionary["conditions"] = {
                            "AF": {"row_N_L_E": 14, "col_N_L_E": 8, "row_p": 14, "col_p": 10},
                            "CHD":{"row_N_L_E": 14, "col_N_L_E": 8, "row_p": 14, "col_p": 10},
                            "HF":{"row_N_L_E": 14, "col_N_L_E": 8, "row_p": 14, "col_p": 10},
                            "HYP":{"row_N_L_E": 14, "col_N_L_E": 8, "row_p": 14, "col_p": 10},
        }   

    elif year == 19:
        
        index_dictionary["conditions"] = {
                            "AF": {"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 10},
                            "CHD":{"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 12},
                            "HF":{"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 10},
                            "HYP":{"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 12},
        }
        
    elif year >= 20:
        
        index_dictionary["conditions"] = {
                            "AF": {"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 10},
                            "CHD":{"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 10},
                            "HF":{"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 10},
                            "HYP":{"row_N_L_E": 13, "col_N_L_E": 8, "row_p": 13, "col_p": 10},
        }
    
    return index_dictionary
   
   
def qof_domain_specifier(year: int) -> str:
    """ Specify domains for qof data.

    This function sets the domain for the qof data for a given year period.
 
    Parameters
    ----------
        year: int
            The starting year of the period.

    Returns
    ----------
        domain: str
            Domain for the dataset.    
    """
    # Set the year period.
    year_period = str(year) + "-" + str(year + 1)

    # Set the base domains.
    base_domain_1 = "https://digital.nhs.uk/data-and-information/publications/statistical/quality-and-outcomes-framework-achievement-prevalence-and-exceptions-data/quality-and-outcomes-framework-qof-20"

    base_domain_2 = "https://digital.nhs.uk/data-and-information/publications/statistical/quality-and-outcomes-framework-achievement-prevalence-and-exceptions-data/20"
    
    # Logic for case handling.
    if year < 17:
        domain = base_domain_1 
    else:
        domain = base_domain_2 

    if year == 18:
        
        # Get the domain from the dataset domain specifier function.
        domain += year_period + "-pas"
        
    else:
        # Get the domain from the dataset domain specifier function.
        domain += year_period
        
    return domain


def qof_extractor(
                    year: int,
                    link: str,
                    qof_dictionary: dict
    ) -> dict :
    """ Extracts qof data for a given year.
    
    This function extracts the relevant data from the qof dataset for a given 
    year.
     
    Parameters
    ----------   
        year: int
            Year to extract data.
        link: str 
            Link for the dataset.
        qof_dictionary: dict
            Dictionary to store the data.   
    
    Returns
    ----------
        qof_dictionary: dict 
            Dictionary to store the data.   
    """
    
    # Initialise intermediate variables. 
    N_L_E = 0
    p = 0

    # Set indices for rows and columns.
    index_dictionary = set_qof_indices(year)
    
    # Loop through conditions.
    for condition_name, condition_data in index_dictionary["conditions"].items():
     
        # Extract indices.
        row_N_L_E = condition_data["row_N_L_E"]
        col_N_L_E = condition_data["col_N_L_E"]
        row_p = condition_data["row_p"]
        col_p = condition_data["col_p"]
      
        # Read in the dataset.
        dataset = pd.read_excel(link, sheet_name=condition_name)
        
        if condition_name=="AF":            
            
            # Store the number of people registered at  GP practice in England.
            N_L_E = dataset.iloc[row_N_L_E, col_N_L_E]
        
        # Store and accumulate the prevalence rates for each condition.
        p += dataset.iloc[row_p, col_p]

    # This data is only avaiable yearly so we need to use at least 2 years.
    qof_dictionary["p_conds_processed"].append(p*0.01)
    qof_dictionary["p_conds_raw"].append(p*0.01)
    qof_dictionary["N_L_E_processed"].append(N_L_E)
    qof_dictionary["N_L_E_raw"].append(N_L_E)

    return qof_dictionary


def qof_looper(parameters: list) -> dict:
    """ Extracts qof data for all requested years.
    
    This function extracts the relevant data from the qof dataset for all
    requested years.
     
    Parameters
    ----------   
        parameters: list
            List of relevant parameters.
    
    Returns
    ----------
        qof_dictionary: dict 
            Dictionary to store the data.
        total_days_counter: int
            Counter tracking the total number of days that data was available.  
    """
    # Unpack parameters
    years = parameters[0]
    browser = parameters[1]
    keywords = parameters[2]

    # Initialise qof dictionary. 
    qof_dictionary = add_data_versions({
                                            "N_L_E": [],
                                            "p_conds": [],
    })

    # Initialise time points for the data interpolation.
    t_points = []

    # Initialise total days counter.
    total_days_counter = 0
    
    # Initialise leap year tracker.
    leap_year_tracker = 0

    # Loop through the years.
    for year_index in range(len(years)):

        try:

            # Get year.
            year = years[year_index]
                
            # Specify the domain.
            domain = qof_domain_specifier(year)
          
            # Access the domain.
            browser.get(domain)
            
            # Add the year to the keywords.
            keywords.append(str(year))

            # Extract the links dataset links from the domain.
            dataset_links = extract_domain_links(browser, keywords, ".xls")
                    
            # Remove the year from the keywords.
            keywords.remove(str(year))

            # Extract the data.
            qof_dictionary = qof_extractor(year, dataset_links[0], qof_dictionary)                    
   

            if leap_year_tracker%4==0:
                
                # Incremenet the number of days counter         
                total_days_counter += 366
    
            else:
                
                # Incremenet the number of days counter         
                total_days_counter += 365
            
            # Add a time point in the middle of the month. 
            t_points.append(total_days_counter)

            
        except:

            break
        
        leap_year_tracker += 1
        
    # Interpolate the data.
    data_dictionary = interpolate_data_dictionary(
                                                    qof_dictionary,
                                                    t_points,
                                                    total_days_counter
    )
    
    return data_dictionary, total_days_counter
        

def extract_qof_data(   
                        browser: selenium.webdriver.chrome.webdriver.WebDriver,
                        collective_start_year: int,
                        collective_end_year: int,
                        no_days: int, 
    ) -> dict:
    """ Manages qof data extraction.
    
    This function manages the extraction of the qof data.
    
    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        no_days: int
            Total number of days to extract data for.    
    Returns
    ----------
        qof_dictionary: dict 
            Dictionary to store the data.   
    """
    # Set start year for qof data.
    start_year = 14
    
    # Set years to search for data.    
    years = list(range(start_year, collective_end_year))
    
    # Set keywords for link extraction.
    keywords = ["reg", "nat"]

    # Set parameters for qof looper.
    parameters = [
                    years,
                    browser,
                    keywords
    ]
    
    # Extract data.
    qof_dictionary, total_days_counter = qof_looper(parameters)


    # Set day indices.
    start_day_index = (start_year - collective_start_year)*365
    end_day_index = start_day_index + total_days_counter

    # Collect data back into the dictionary.
    qof_dictionary = reindex_data(start_day_index, end_day_index, no_days, qof_dictionary)

    return qof_dictionary


def interpolate_booking_delays(
                                daily_booking_delays: list,
                                max_delay: int
    ) -> list:
    """ Interpolates the booking delays for the primary care data. 
    
    This function evenly distributes the delays for each time period 
    into each time within the period. 

    Example:

    Time period = [2, 7]
    Number of appointments with a delay in the time period = 70

    This function produces a list for this time period 

    [10, 10, 10, 10, 10, 10] 

    which represents 10 appointments with a delay of 2 days, 10 appointments with 
    a delay of 3 days etc. 

    The time periods are:

    [0], [1], [2-7], [8-14], [15-21], [22-28], [28-max_delay]
     
    Parameters
    ----------   
        daily_booking_delays: list 
            Array of the number of appointments with a given delay between 
            booking and appointment date. 
        max_delay: int
            Upper bound on the delay between booking and having an appointment.
    
    Returns
    ----------
        booking_delays: list 
            List with the booking delays for days in [0, max_delay]. 
    """
    # Same day appointments.
    b_0 = daily_booking_delays[0]
    
    # Next day appointments.
    b_1 = daily_booking_delays[1]
    
    # Store these in a list. 
    booking_delays = [b_0, b_1]

    # Partition the number of appointments for each interval 
    # equally for each delay range.. 
     
    # [2-7] 
    for i in range(0, 6):
        booking_delays.append(daily_booking_delays[2]/6)

    # [8-14] 
    for i in range(0, 7):
        booking_delays.append(daily_booking_delays[3]/7)

    # [15-21] 
    for i in range(0, 7):
        booking_delays.append(daily_booking_delays[4]/7)

    # [22-28] 
    for i in range(0, 7):
        booking_delays.append(daily_booking_delays[5]/7)
    
    # [28-max_delay] 
    for i in range(0, max_delay - 28):
        booking_delays.append(daily_booking_delays[6]/(max_delay - 28))

    return booking_delays


def primary_care_domain_specifier(year: int, month: str) -> str:
    """ Specify domains for the primary care data.

    This function sets the domain for the primary care data for a given year period.
 
    Parameters
    ----------
        year : int
            The starting year of the period.
        month : str
            The month of the period.
    Returns
    ----------
        domain : str
            Domain for the dataset.    
    """
    # Exception handling. 
    if year == 19 and month == "july":
        month = "jul"

    # Set the time period.
    time_period = month + "-20" + str(year)

    # Base domain.
    base_domain = "https://digital.nhs.uk/data-and-information/publications/statistical/appointments-in-general-practice/"
    
    # Set domain.
    domain = base_domain + time_period 
        
    return domain


def primary_care_extractor(
                            link:str,
                            month_length: int,
                            total_days_counter: int,
                            start_day_index: int,
                            B_delay: np.array,
                            max_delay: int, 
                            pc_dictionary: dict
    ):
    """ Extracts primary care data for a given month.
    
    This function extracts the primary care data from the primary care dataset
    for a given month.
     
    Parameters
    ----------   
        link: str 
            Link for the dataset.  
        month_length: int,
            Number of days in given month.
        total_days_counter: int
            Counter for the total number of days the data has been extracted for.
        start_day_index: int
            Starting date of data extraction.
        B_delay: np.array            
            Array to store the number of appointments that occured on the day 
            with a delay of k days for k in [0, max_delay].
        max_delay: int
            Upper bound on the delay between booking and having an appointment.
        pc_dictionary: dict
            Dictionary to store the data.   
    
    Returns
    ----------
        B_delay: np.array
            Array to store TODO 
        pc_dictionary: dict 
            Dictionary to store the data.   
    """
    # Read in the dataset.
    dataset = pd.read_excel(link, sheet_name="Table 2d")

    daily_appointments_in_month = []
    
    for day_index in range(month_length):
        
        # Extract the number of daily appointments. 
        daily_appointments = dataset.iloc[12 + day_index, 2]

        # Update the data dictionary. 
        pc_dictionary["A_GP_processed"].append(daily_appointments)

        daily_appointments_in_month.append(daily_appointments)
        
    # Update the data dictionary. 
    pc_dictionary["A_GP_raw"].append(np.nanmean(daily_appointments_in_month))
    
    # Compute the average delays across the month.
    avg_delays = np.mean(dataset.iloc[12:12 + month_length, 4:11], axis=0)

    # Update the delay array.
    B_delay[start_day_index+total_days_counter:start_day_index+total_days_counter+month_length, :] = interpolate_booking_delays(avg_delays, max_delay)
    
    return B_delay, pc_dictionary


def primary_care_looper(parameters: list):
    """ Extracts primary care data for all requested years.
    
    This function extracts the relevant data from the primary care dataset for all
    requested years.
     
    Parameters
    ----------   
        parameters: list
            List of relevant parameters.
    
    Returns
    ----------
        pc_dictionary: dict 
            Dictionary to store the data.   
    """
    # Unpack parameters
    years = parameters[0]
    keywords = parameters[1]
    browser = parameters[2]
    B_delay = parameters[3]
    max_delay = parameters[4]
    start_day_index = parameters[5]

    # Initialise data dictionary.
    pc_dictionary = add_data_versions({"A_GP": [], "B_GP": []}) 
    
    # Initialise time point storage for interpolation.
    t_points = []

    # Initialise total days counter.
    total_days_counter = 0
    
    # Initialise leap year tracker.
    leap_year_tracker = 0 
    
    # Loop through the years.
    for year in years:
        
        # Loop through the months.
        for month_index in range(12):
            
            try:
                # Specify the domain
                domain = primary_care_domain_specifier(year, months[month_index])
            
                # Access the domain.
                browser.get(domain)

                # Extract the links for the dataset.
                dataset_links = extract_domain_links(browser, keywords, ".xls")
                    
                # Look up month length.
                month_length = month_lengths[month_index]
                    
                # If it's a leap year then add a day. 
                if leap_year_tracker%4 == 0:
                    if month_index==1:
                        month_length = 29

                # Extract the data. 
                B_delay, pc_dictionary = primary_care_extractor(    
                                                                dataset_links[0],
                                                                month_length,
                                                                total_days_counter,
                                                                start_day_index,
                                                                B_delay,
                                                                max_delay,
                                                                pc_dictionary
                )
                    
                # Update the total day index counter.         
                total_days_counter += month_length

                # Add a time point in the middle of the month. 
                t_points.append(total_days_counter)
                    
            except: 
                pass
        
        leap_year_tracker += 1

    # Compute the bookings.
    pc_dictionary =  update_bookings(
                                        start_day_index,
                                        total_days_counter-month_length,
                                        max_delay,
                                        B_delay,
                                        pc_dictionary
    )
    
    pc_dictionary["A_GP_raw"] = interpolate_data(
                                                pc_dictionary["A_GP_raw"],
                                                t_points,
                                                total_days_counter,
                                                "zeros"
    )
    
    return pc_dictionary, total_days_counter


#### TODO - This needs more explanation. 
def update_bookings(
                        start_day_index: int,
                        total_days_counter: int,
                        max_delay: int,
                        B_delay: np.array,
                        pc_dictionary: dict
    ):
    """ Updates the primary care bookings data.
    
    This function updates the primary care booking data. 
     
    Parameters
    ----------   
        start_day_index: int
            Starting date of data extraction.
        total_day_index: int
            Total days of data extraction index. 
        max_delay: int
            Upper bound on the delay between booking and having an appointment.
        B_delay: np.array
            Array to store the number of appointments that occured on the day 
            with a delay of k days for k in [0, max_delay].
        pc_dictionary: dict
            Dictionary to store the data.   
            
    Returns
    ----------
        pc_dictionary: dict 
            Dictionary to store the data.   
    """
    B_delay_length = B_delay.shape[0]
  
    # Update the primary care dictionary with the booking data.  
    for i in range(start_day_index, start_day_index+total_days_counter-max_delay):
        B_GP = 0
        for j in range(max_delay):
            if i+j < B_delay_length:
                B_GP += B_delay[i+j, j]

        pc_dictionary["B_GP_processed"].append(B_GP)
        pc_dictionary["B_GP_raw"].append(B_GP)

    return pc_dictionary


def extract_primary_care_data(      
                                browser: selenium.webdriver.chrome.webdriver.WebDriver,
                                collective_start_year: int,
                                collective_end_year: int,
                                no_days: int,
                                max_delay: int
    ):
    """ Manages primary care data extraction.
    
    This function manages the extraction of the primary care data. 

    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        no_days: int
            Total number of days to extract data for. 
        max_delay: int
            Upper bound on the delay between booking and having an appointment.
    Returns
    ----------
        pc_dictionary: dict 
            Dictionary to store the data.   
    """
    # Set start year for primary care data.
    start_year = 19
    
    # Increment max delay for indexing purposes. 
    max_delay = max_delay + 1
    
    # Set years to search for data.    
    years = list(range(start_year, collective_end_year+1))

    # Initialise arrays.
    B_delay = np.zeros((no_days, max_delay+1))

    # Set the keywords for link extraction.
    keywords = ["GP", "Publication"]

    # Set the start day index. 
    start_day_index = (start_year - collective_start_year)*365

    # Set parameters for primary care looper. 
    parameters = [
                    years,
                    keywords,
                    browser,
                    B_delay,
                    max_delay,
                    start_day_index,
    ]

    # Extract the data.
    pc_dictionary, total_days_counter = primary_care_looper(parameters)

    # Set the end day index. 
    end_day_index = start_day_index + total_days_counter

    # Collect data back in dictionary.
    pc_dictionary = reindex_data(start_day_index, end_day_index, no_days, pc_dictionary, max_delay)  
    
    return pc_dictionary

## This might not make sense. The way I've written the loopers in the 
## extraction functions means that sometimes the 
def resolve_index_discrepancy(
                                start_day_index: int,
                                end_day_index: int,
                                no_days: int,
                                data: np.array):
    """

    This function deals with index discrepancies from the data
    extraction step. 
        
    Parameters
    ----------           
        start_day_index: int
            Index of first day data was available from that dataset.
        end_day_index: int
            Index of final day data was available from that dataset.
        no_days: int
            Total number of days to extract data for. 
        data: np.array
            Data array.
  
    Returns
    ----------
        data_array: np.array 
            Corrected data
    """        
    # Initialise array that is as long as the total number of days.
    data_array = np.zeros((no_days))
   
    # Number of days
    no_days_collected = len(data)
    no_available_days = len(data_array[start_day_index:end_day_index])
    

    if no_days_collected == no_available_days:
        data_array[start_day_index:end_day_index] = data
    if no_days_collected > no_available_days:
        data_array[start_day_index:end_day_index] = data[0:no_available_days]
    else:
        data_array[start_day_index:start_day_index+no_days_collected] = data

    return data_array


def compute_diagnostic_proportions():
    """ Function to compute diagnostic test proportions.
    
    This function uses a dataset provided by Mike Woodall at NHS Strategy to 
    approximate the proportion of MRI and CT diagnostic tests that are performed
    for cardiology conditions.

    Returns
    ----------
        alpha_MRI_cardiology : float 
            The proportion of MRI tests for cardiology conditions.
        alpha_CT_cardiology : float
            The proportion of CT tests for cardiology conditions.
    """
    # Set the path for the dataset.
    diagnostic_proportions_path = os.getcwd() + "/diagnostic_proportions.csv"
    
    # Read in the dataset.
    diagnostic_proportions_df = pd.read_csv(diagnostic_proportions_path)

    # Set conditions.
    MRI_condition = diagnostic_proportions_df["test_name"] == "Magnetic Resonance Imaging"
    CT_condition = diagnostic_proportions_df["test_name"] == "Computed Tomography"
    codes_condition = [x in codes for x in diagnostic_proportions_df["treatment_function"]]

    # Compute the proportions.
    alpha_MRI_cardiology =  np.sum(diagnostic_proportions_df["n"][np.logical_and(MRI_condition, codes_condition)]) \
                                / np.sum(diagnostic_proportions_df["n"][MRI_condition])
    
    alpha_CT_cardiology = np.sum(diagnostic_proportions_df["n"][np.logical_and(CT_condition, codes_condition)]) \
                            / np.sum(diagnostic_proportions_df["n"][CT_condition])


    return alpha_MRI_cardiology, alpha_CT_cardiology


def diagnostics_domain_specifier(year: int) -> str:
    """ Specify domains for the diagnostics data.

    This function sets the domain for the diagnostic data for a given year period.
 
    Parameters
    ----------
        year : int
            The starting year of the period.
    Returns
    ----------
        domain : str
            Domain for the dataset.    
    """
    # Set the year period.
    year_period = str(year) + "-" + str(year + 1) + "/"
    
    # Base domain.
    base_domain = "https://www.england.nhs.uk/statistics/statistical-work-areas/diagnostics-waiting-times-and-activity/monthly-diagnostics-waiting-times-and-activity/monthly-diagnostics-data-20"

    # Set domain.
    domain = base_domain + year_period 
        
    return domain


def diagnostics_extractor(
                            link: str,
                            month_index: int, 
                            alpha_MRI_cardiology: float,
                            alpha_CT_cardiology: float,
                            diagnostics_dictionary: dict,
    ) -> dict:
    """ Extracts diagnostics data for a given month.
    
    This function extracts the relevant data from the diagnostics dataset for
    a given month.
     
    Parameters
    ----------   
        link: str 
            Link for the dataset.
        month_index
            Index for the requested month. 
        alpha_MRI_cardiology : float 
            The proportion of MRI tests for cardiology conditions.
        alpha_CT_cardiology : float
            The proportion of CT tests for cardiology conditions.
        diagnostics_dictionary: dict
            Dictionary to store the data.   
            
    Returns
    ----------
        diagnostics_dictionary: dict 
            Dictionary to store the data.   
    """
    # Read in the dataset.
    dataset = pd.read_excel(link, sheet_name="National")

    # Determine month length.
    month_length = month_lengths[month_index]

    # Extract the number of diagnostic tests performed during the month
    # and partition the monthly tests into a constant number of daily
    # tests throughout the month.  
    N_MRI_cardiology = (alpha_MRI_cardiology/month_length) \
                            *(dataset.iloc[15, 22] + dataset.iloc[15, 24])

    N_CT_cardiology = (alpha_CT_cardiology/month_length) \
                            *(dataset.iloc[16, 22] + dataset.iloc[16, 24])

    N_echocardiography = (1/month_length) \
                        *(dataset.iloc[21, 22] + dataset.iloc[21, 24])

    N_electrophysiology = (1/month_length) \
                            *(dataset.iloc[22, 22] + dataset.iloc[22, 24])

    # Store the activity data. 
    diagnostics_dictionary["N_MRI_cardiology_processed"].extend([N_MRI_cardiology]*month_length)
    diagnostics_dictionary["N_CT_cardiology_processed"].extend([N_CT_cardiology]*month_length)
    diagnostics_dictionary["N_echocardiography_processed"].extend([N_echocardiography]*month_length)
    diagnostics_dictionary["N_electrophysiology_processed"].extend([N_electrophysiology]*month_length)

    diagnostics_dictionary["N_MRI_cardiology_raw"].append(N_MRI_cardiology*month_length)
    diagnostics_dictionary["N_CT_cardiology_raw"].append(N_CT_cardiology*month_length)
    diagnostics_dictionary["N_echocardiography_raw"].append(N_echocardiography*month_length)
    diagnostics_dictionary["N_electrophysiology_raw"].append(N_electrophysiology*month_length)

    # Extract the diagnostic waiting lists.
    W_MRI_cardiology = alpha_MRI_cardiology*dataset.iloc[15, 3]
    
    W_CT_cardiology = alpha_CT_cardiology*dataset.iloc[16, 3]
    
    W_echocardiography = dataset.iloc[21, 3]

    W_electrophysiology = dataset.iloc[22, 3]

    # Store the waiting list data. 
    diagnostics_dictionary["W_MRI_cardiology_processed"].append(W_MRI_cardiology)
    diagnostics_dictionary["W_CT_cardiology_processed"].append(W_CT_cardiology)
    diagnostics_dictionary["W_echocardiography_processed"].append(W_echocardiography)
    diagnostics_dictionary["W_electrophysiology_processed"].append(W_electrophysiology)

    diagnostics_dictionary["W_MRI_cardiology_raw"].append(W_MRI_cardiology)
    diagnostics_dictionary["W_CT_cardiology_raw"].append(W_CT_cardiology)
    diagnostics_dictionary["W_echocardiography_raw"].append(W_echocardiography)
    diagnostics_dictionary["W_electrophysiology_raw"].append(W_electrophysiology)

    return diagnostics_dictionary


def diagnostics_looper(parameters: list):
    """ Extracts diagnostic data for all requested years.
    
    This function extracts the relevant data from the diagnostics dataaset for all
    requested years.
     
    Parameters
    ----------   
        parameters: list
            List of relevant parameters.
    
    Returns
    ----------
        diagnostics_dictionary: dict 
            Dictionary to store the data.   
        total_days_counter: int
            Counter for the total number of days the data has been extracted for.
    """
    # Unpack parameters.
    years = parameters[0]
    keywords = parameters[1]
    browser = parameters[2]
    alpha_MRI_cardiology = parameters[3]
    alpha_CT_cardiology = parameters[4]

    # Initialise dataset links list. 
    dataset_links = []

    # Initialise data dictionary.
    diagnostics_dictionary = add_data_versions({
                                "W_MRI_cardiology" : [],
                                "W_CT_cardiology": [],
                                "W_echocardiography" : [],
                                "W_electrophysiology" : [],
                                "N_MRI_cardiology" : [],
                                "N_CT_cardiology" : [],
                                "N_echocardiography" : [],
                                "N_electrophysiology" : []
    })
    
    years.insert(0, 14)
    
    # Loop through the requested years. 
    for year in years:

        # Specify the domain. 
        domain = diagnostics_domain_specifier(year)

        # Access the domain. 
        browser.get(domain)

        # Extract the dataset links and add them to the list. 
        dataset_links.extend(extract_domain_links(browser, keywords, ".xls"))

    # We only want the data from 2015 onwards.
    dataset_links = [link for link in dataset_links if "2014.xls" not in link]

    # Initialise time point storage for interpolation.
    t_points = []

    # Initialise total months counter. 
    total_months_counter = 0
    
    # Initialise total days counter. 
    total_days_counter = 0

    # Loop through all the dataset links.
    for link in dataset_links:
        
        # Compute the month index. 
        month_index = total_months_counter % 12
    
        # Extract the data. 
        diagnostics_dictionary = diagnostics_extractor(
                                                        link,
                                                        month_index,
                                                        alpha_MRI_cardiology,
                                                        alpha_CT_cardiology,
                                                        diagnostics_dictionary
        )                    
        


        # Look up month length.
        month_length = month_lengths[month_index]
                
        # If it's a leap year then add a day. 
        if total_months_counter%48 == 0:
            if month_index==1:
                month_length = 29
                        

        # Incremenet the number of days counter 
        total_days_counter += month_length
        
        # Increment the total months counter.        
        total_months_counter += 1    

        # Add a time point at the beginning of the month. 
        t_points.append(total_days_counter)
        
    # Set the ignore data labels list as we don't need to interpolate everything.
    ignore_data_labels = [  
                            "N_MRI_cardiology_processed",
                            "N_CT_cardiology_processed",
                            "N_echocardiography_processed",
                            "N_electrophysiology_processed",
    ]   
    
    # Interpolate data.                                                                                                        
    diagnostics_dictionary = interpolate_data_dictionary(
                                                            diagnostics_dictionary, 
                                                            t_points,
                                                            total_days_counter,
                                                            ignore_data_labels
    )    

    return diagnostics_dictionary, total_days_counter

         
def extract_diagnostics_data(      
                                browser: selenium.webdriver.chrome.webdriver.WebDriver,
                                collective_start_year: int,
                                collective_end_year: int,
                                no_days: int, 
    ) -> dict:
    """ Manages diagnostics data extraction.
    
    This function manages the extraction of the diagnostic data.

    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        no_days: int
            Total number of days to extract data for. 
 
    Returns
    ----------
        diagnostics_dictionary: dict 
            Dictionary to store the data.  
    """
    # Set start year for diagnostics data.
    start_year = 15
    
    # Set years to search for data.    
    years = list(range(start_year, collective_end_year+1))

    # Compute MRI and CT proportions from proportions dataset.
    alpha_MRI_cardiology, alpha_CT_cardiology = compute_diagnostic_proportions()

    # Set keywords for link extraction.
    keywords = ["Commissioner-"]

    # Set parameters for diagnostics looper.
    parameters = [
                    years,
                    keywords,
                    browser,
                    alpha_MRI_cardiology,
                    alpha_CT_cardiology
    ] 
    
    # Extract the data.
    diagnostics_dictionary, total_days_counter = diagnostics_looper(parameters)

    # Set start and end indices.
    start_day_index = (start_year - collective_start_year)*365
    end_day_index = start_day_index + total_days_counter

    # Collect data back into dictionary.
    diagnostics_dictionary = reindex_data(start_day_index, end_day_index, no_days, diagnostics_dictionary)  
    
    return diagnostics_dictionary


def hes_domain_specifier(browser: selenium.webdriver.chrome.webdriver.WebDriver) -> str:
    """ Specify domain for the hes data.

    This function specifies the domain for the hes data.
 
    Parameters
    ----------
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
            
    Returns
    ----------
        domain : str
            Domain for the dataset.     
    """
    # Specify the initial domain. 
    initial_domain = "https://digital.nhs.uk/data-and-information/publications/statistical/hospital-episode-statistics-for-admitted-patient-care-outpatient-and-accident-and-emergency-data"

    # Access the initial domain. 
    browser.get(initial_domain)
    
    # Extract the elements. 
    elements = browser.find_elements_by_xpath('//a[contains(@href, "")]')
        
    # Extract the links.
    links = [element.get_attribute("href") for element in elements]
    
    # Extract the domain link for the datasets. 
    data_domain = "/data-and-information/publications/statistical/hospital-episode-statistics-for-admitted-patient-care-outpatient-and-accident-and-emergency-data/"
    
    domain = [link for link in links if data_domain in link][0]
    
    return domain


def hes_extractor(
                    month_index: int,
                    leap_year_tracker: int,
                    year: int, 
                    total_days_counter: int,
                    t_points: list,
                    dataset: pd.DataFrame,
                    hes_dictionary: dict
    ):
    """ Extracts hes data for a given month.
    
    This function extracts the relevant data from the hes dataset for
    a given month.
     
    Parameters
    ----------   
        month_index: int
            Index for the requested month. 
        year: int
            Requested year.       
        total_days_counter: int
            Counter to track the total number of days that the data has
            been extracted for.   
        t_points: list
            List of associated time points for the data.
        hes_dictionary: dict
            Dictionary to store the data.   
            
    Returns
    ----------
        hes_dictionary: dict 
            Dictionary to store the data.   
        total_days_counter: int
            Counter for the total number of days the data has been extracted for.    
        t_points: list
            List of associated time points for the data.                    
    """
    # Determine the period name to search through the dataframe.
    period_name = months_hes[month_index] + str(year)
            
    # Extract indices in the dataframe with this period name.
    period_indices = dataset["Month_Ending"] == period_name

    # Exception handling.
    if np.sum(period_indices) != 0:

        # Find indices for the specialties in codes. 
        codes_indces = dataset["TRETSPEF"].isin(codes)
        
        # Find indices matching period and specialties. 
        combined_indices = np.logical_and(period_indices, codes_indces)

        # Compute the length of the month. 
        month_length = month_lengths[month_index]
        
    
        # If it's a leap year then add a day. 
        if leap_year_tracker%4 == 0:
            if month_index==1:
                month_length = 29

        # Extract monthly totals and then partition this equally for each day 
        # in the month.  
        N_FCE_codes = (1/month_length)*dataset[combined_indices]["FCE"].sum()
        N_FCEP_codes = (1/month_length)*dataset[combined_indices]["FCEs_With_Procedure"].sum()
        N_A_codes = (1/month_length)*dataset[combined_indices]["Attended_Appointments"].sum()                
        
        # Store data.
        hes_dictionary["N_FCE_codes_processed"].extend([N_FCE_codes]*month_length)
        hes_dictionary["N_FCEP_codes_processed"].extend([N_FCEP_codes]*month_length)
        hes_dictionary["N_A_codes_processed"].extend([N_A_codes]*month_length)

        hes_dictionary["N_FCE_codes_raw"].append(N_FCE_codes*month_length)
        hes_dictionary["N_FCEP_codes_raw"].append(N_FCEP_codes*month_length)
        hes_dictionary["N_A_codes_raw"].append(N_A_codes*month_length)

        # Incremenet the number of days counter.
        total_days_counter += month_lengths[month_index]

        # Add a time point at the beginning of the month.
        t_points.append(total_days_counter)
        
    
    return hes_dictionary, total_days_counter, t_points


def hes_looper(parameters: list):
    """ Extracts hes data for all requested years.
    
    This function extracts the relevant data from the hes dataset for all
    requested years.
     
    Parameters
    ----------   
        parameters: list
            List of relevant parameters.
    
    Returns
    ----------
        hes_dictionary: dict 
            Dictionary to store the data.   
        total_days_counter: int
            Counter for the total number of days the data has been extracted for.
    """
    # Unpack parameters.
    years = parameters[0]
    keywords = parameters[1]
    browser = parameters[2]
    
    # Initialise dictionary.
    hes_dictionary = add_data_versions({
                        "N_FCE_codes" : [],
                        "N_FCEP_codes" : [],
                        "N_A_codes" : []
    })
    
    # Specify the domain.
    domain = hes_domain_specifier(browser)

    # Access the domain.
    browser.get(domain)
    
    # Extract the dataset link.
    dataset_link = extract_domain_links(browser, keywords, ".csv")[0]

    # Initialise total days counter.
    total_days_counter = 0

    # Initialise time point storage for interpolation.
    t_points = []
        
    # Read in the dataset.
    dataset = pd.read_csv(dataset_link)

    # Initialise loop year tracker. 
    leap_year_tracker = 0
    
    # Loop through the years.
    for year in years:

        # Loop through the months. 
        for month_index in range(12):
            
            # Extract data. 
            hes_dictionary, total_days_counter, t_points = hes_extractor(
                                                        month_index, 
                                                        leap_year_tracker,
                                                        year,
                                                        total_days_counter,
                                                        t_points, 
                                                        dataset,
                                                        hes_dictionary
            )
            
    # Set the ignore data labels list as we don't need to interpolate everything.
    ignore_data_labels = [  
                            "N_FCE_codes_processed",
                            "N_FCEP_codes_processed",
                            "N_A_codes_processed",
    ]    

    # Interpolate data.                                                                                                        
    hes_dictionary = interpolate_data_dictionary(
                                                    hes_dictionary, 
                                                    t_points,
                                                    total_days_counter,
                                                    ignore_data_labels
    )    

    return hes_dictionary, total_days_counter


def extract_hes_data(     
                        browser: selenium.webdriver.chrome.webdriver.WebDriver,
                        collective_start_year: int,
                        collective_end_year: int,
                        no_days: int, 
    ) -> dict:
    """ Manages hes data extraction.
    
    This function manages the extraction of the hes data.
    
     
    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        no_days: int
            Total number of days to extract data for. 
 
    Returns
    ----------
        hes_dictionary: dict 
            Dictionary to store the data.   
    """
    # Set start year for diagnostics data.
    start_year = 18
    
    # Set years to search for data.    
    years = list(range(start_year, collective_end_year+1))
    
    # Set keywords for link extraction.
    keywords = ["TREATMENT", "SPECIALTY"]

    # Set parameters for hes looper.
    parameters = [years, keywords, browser] 

    # Extract the data.
    hes_dictionary, total_days_counter = hes_looper(parameters)
    
    # Set start and end indices.
    # Note: HES data starts from March 2018
    start_day_index = (start_year - collective_start_year)*365 + np.sum(month_lengths[0:4])
    end_day_index = start_day_index + total_days_counter
    
    # Collect data back into dictionary.
    hes_dictionary = reindex_data(start_day_index, end_day_index, no_days, hes_dictionary)  
    
    return hes_dictionary


def consultant_rtt_domain_specifier(year: int) -> str:
    """ Specify domains for the consultant rtt data.

    This function sets the domain for the consultant rtt data for a given year period.
 
    Parameters
    ----------
        year: int
            The starting year of the period.
            
    Returns
    ----------
        domain : str
            Domain for the dataset.     
    """
    # Set the year period.
    year_period = str(year) + "-" + str(year + 1) + "/"
    
    # Base domain.
    base_domain = "https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/rtt-data-20"

    # Set domain.
    domain = base_domain + year_period 
        
    return domain


def consultant_rtt_extractor( 
                                incomplete_dataset_link: str,
                                complete_admitted_dataset_link: str,
                                complete_nonadmitted_dataset_link: str,
                                new_dataset_link: str,
                                consultant_rtt_dictionary: dict,
                                month_index: int
    ) -> dict:                                
    """ Extracts consultant rtt for a given month.
    
    This function extracts the relevant data from the consultant rtt dataset for
    a given month.
     
    Parameters
    ----------                                  
        incomplete_dataset_link: str
            Link for the incomplete consultant rtt dataset.                                
        complete_admitted_dataset_link: str
            Link for the complete admitted consultant rtt dataset.
        complete_nonadmitted_dataset_link: str
            Link for the complete non-admitted consultant rtt dataset.                                                                                            
        new_dataset_link: str
            Link for the new consultant rtt dataset.        
        consultant_rtt_dictionary: dict
            Dictionary to store the data.   
        month_index: int
            Index for the requested month. 
            
    Returns
    ----------
        consultant_rtt_dictionary: dict
            Dictionary to store the data.   
    """
    # Determine month length.
    month_length = month_lengths[month_index]

    # Read in the datasets.

    incomplete_dataset = pd.read_excel(incomplete_dataset_link, sheet_name="National")
    incomplete_dataset_dta = pd.read_excel(incomplete_dataset_link, sheet_name="National with DTA")
    complete_admitted_dataset = pd.read_excel(complete_admitted_dataset_link, sheet_name="National")
    complete_nonadmitted_dataset = pd.read_excel(complete_nonadmitted_dataset_link, sheet_name="National")
    new_dataset = pd.read_excel(new_dataset_link, sheet_name="National")

    # Set the base index for the spreadsheets. 
    base_index = 56

    # Change the baseindex for these anomalies.
    if ("21-" in incomplete_dataset_link and month_index >= 3) or "22-" in incomplete_dataset_link:

        base_index += 52
    
    # Extract the waiting list data. 
    W_incomplete_cs = incomplete_dataset.iloc[21, base_index]
    W_incomplete_c = incomplete_dataset.iloc[24, base_index]
    W_incomplete_dta_cs = incomplete_dataset_dta.iloc[21, base_index]
    W_incomplete_dta_c = incomplete_dataset_dta.iloc[24, base_index]
    
    # Store the waiting list data. 
    consultant_rtt_dictionary["W_incomplete_cs_processed"].append(W_incomplete_cs)
    consultant_rtt_dictionary["W_incomplete_c_processed"].append(W_incomplete_c)
    consultant_rtt_dictionary["W_incomplete_dta_cs_processed"].append(W_incomplete_dta_cs)
    consultant_rtt_dictionary["W_incomplete_dta_c_processed"].append(W_incomplete_dta_c)
    
    consultant_rtt_dictionary["W_incomplete_cs_raw"].append(W_incomplete_cs)
    consultant_rtt_dictionary["W_incomplete_c_raw"].append(W_incomplete_c)
    consultant_rtt_dictionary["W_incomplete_dta_cs_raw"].append(W_incomplete_dta_cs)
    consultant_rtt_dictionary["W_incomplete_dta_c_raw"].append(W_incomplete_dta_c)
    
    # Extract the activity data.
    N_completed_admitted_cs= (1/month_length)*complete_admitted_dataset.iloc[21, base_index+1]
    N_completed_admitted_c = (1/month_length)*complete_admitted_dataset.iloc[24, base_index+1]
    N_completed_nonadmitted_cs = (1/month_length)*complete_nonadmitted_dataset.iloc[21, base_index+1]
    N_completed_nonadmitted_c = (1/month_length)*complete_nonadmitted_dataset.iloc[24, base_index+1]
    N_new_cs = (1/month_length)*new_dataset.iloc[21, 3]
    N_new_c = (1/month_length)*new_dataset.iloc[24, 3]

    # Store the activity data. 
    consultant_rtt_dictionary["N_completed_admitted_cs_processed"].extend([N_completed_admitted_cs]*month_length)
    consultant_rtt_dictionary["N_completed_admitted_c_processed"].extend([N_completed_admitted_c]*month_length)
    consultant_rtt_dictionary["N_completed_nonadmitted_cs_processed"].extend([N_completed_nonadmitted_cs]*month_length)
    consultant_rtt_dictionary["N_completed_nonadmitted_c_processed"].extend([N_completed_nonadmitted_c]*month_length)
    consultant_rtt_dictionary["N_new_cs_processed"].extend([N_new_cs]*month_length)
    consultant_rtt_dictionary["N_new_c_processed"].extend([N_new_c]*month_length)

    consultant_rtt_dictionary["N_completed_admitted_cs_raw"].append(N_completed_admitted_cs*month_length)
    consultant_rtt_dictionary["N_completed_admitted_c_raw"].append(N_completed_admitted_c*month_length)
    consultant_rtt_dictionary["N_completed_nonadmitted_cs_raw"].append(N_completed_nonadmitted_cs*month_length)
    consultant_rtt_dictionary["N_completed_nonadmitted_c_raw"].append(N_completed_nonadmitted_c*month_length)
    consultant_rtt_dictionary["N_new_cs_raw"].append(N_new_cs*month_length)
    consultant_rtt_dictionary["N_new_c_raw"].append(N_new_c*month_length)

    return consultant_rtt_dictionary


def consultant_rtt_looper(parameters: list):
    """ Extracts consultant rtt data for all requested years.
    
    This function extracts the relevant data from the consultant rtt dataaset for all
    requested years.
     
    Parameters
    ----------   
        parameters: list
            List of relevant parameters.
    
    Returns
    ----------
        consultant_rtt_dictionary: dict 
            Dictionary to store the data.   
        total_days_counter: int
            Counter for the total number of days the data has been extracted for.
    """
    # Unpack parameters.
    years = parameters[0]
    keywords = parameters[1]
    browser = parameters[2]
    
    # Initialise dataset links list.    
    dataset_links = []
 
    # Initialise data dictionary.
    consultant_rtt_dictionary = add_data_versions({
                                "W_incomplete_cs" : [],
                                "W_incomplete_c": [],
                                "W_incomplete_dta_cs" : [],
                                "W_incomplete_dta_c" : [],
                                "N_completed_admitted_cs" : [],
                                "N_completed_admitted_c" : [],
                                "N_completed_nonadmitted_cs" : [],
                                "N_completed_nonadmitted_c" : [],
                                "N_new_cs" : [],
                                "N_new_c" : []                                
    })
 
    years.insert(0, 16)
    
    # Loop through the requested years. 
    for year in years:
        
        # Specify the domain. 
        domain = consultant_rtt_domain_specifier(year)

        # Access the domain. 
        browser.get(domain)
        
        # Extract the dataset links and add them to the list. 
        dataset_links.extend(extract_domain_links(browser, [], ".xls"))

    # We only want the data from 2017 onwards.
    dataset_links = [link for link in dataset_links if "16-" not in link]

    # Extract the links using the keywords.
    incomplete_dataset_links = unique_list([x for x in dataset_links if keywords[0] in x])

    complete_admitted_dataset_links = unique_list([x for x in dataset_links if keywords[1] in x
                                                    and keywords[2] not in x])
    
    complete_nonadmitted_dataset_links = unique_list([x for x in dataset_links if keywords[2] in x])
    
    new_dataset_links = unique_list([x for x in dataset_links if keywords[3] in x])

    # Store the number of links.
    no_links = len(incomplete_dataset_links)

    # Initialise time point storage for interpolation.
    t_points = []
    
    # Initialise total months counter. 
    total_months_counter = 0

    # Initialise total days counter. 
    total_days_counter = 0

    # Loop through all the dataset links.
    for link_index in range(no_links):

        # Compute the month index. 
        month_index = total_months_counter % 12
    
        # Extract the data. 
        incomplete_dataset_link = incomplete_dataset_links[link_index] 
        complete_admitted_dataset_link = complete_admitted_dataset_links[link_index]
        complete_nonadmitted_dataset_link = complete_nonadmitted_dataset_links[link_index]
        new_dataset_link = new_dataset_links[link_index]

        consultant_rtt_dictionary = consultant_rtt_extractor(
                                                    incomplete_dataset_link,
                                                    complete_admitted_dataset_link,
                                                    complete_nonadmitted_dataset_link,
                                                    new_dataset_link,
                                                    consultant_rtt_dictionary,
                                                    month_index
        )  

        # Look up month length.
        month_length = month_lengths[month_index]
       
        # If it's a leap year then add a day. 
        if total_months_counter%48 == 0:
            if month_index==1:
                month_length = 29                        
        
        # Increment the number of days counter.
        total_days_counter += month_length

        # Increment the total months counter.        
        total_months_counter += 1    

        # Add a time point at the beginning of the month.
        t_points.append(total_days_counter)
    
    # Set the ignore data labels list as we don't need to interpolate everything.
    ignore_data_labels = [  
                            "N_completed_admitted_cs_processed",
                            "N_completed_admitted_c_processed",
                            "N_completed_nonadmitted_cs_processed",
                            "N_completed_nonadmitted_c_processed",
                            "N_new_cs_processed",
                            "N_new_c_processed",
    ]     
    
    # Interpolate data.                                                                                                        
    consultant_rtt_dictionary = interpolate_data_dictionary(
                                                            consultant_rtt_dictionary, 
                                                            t_points,
                                                            total_days_counter,
                                                            ignore_data_labels
    )    

    return consultant_rtt_dictionary, total_days_counter


def extract_consultant_rtt_data(
                                    browser: selenium.webdriver.chrome.webdriver.WebDriver,
                                    collective_start_year: int,
                                    collective_end_year: int,
                                    no_days: int, 
    ) -> dict:
    """ Manages consultant rtt data extraction.
    
    This function manages the extraction of the consultant rtt data.
    
     
    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        no_days: int
            Total number of days to extract data for. 

    Returns
    ----------
        consultant_rtt_dictionary: dict
            Dictionary to store the data.   
    """
    # Set start year for diagnostics data.
    start_year = 17
    
    # Set years to search for data.    
    years = list(range(start_year, collective_end_year+1))

    # Set keywords for link extraction.
    keywords = [
                    "Incomplete-Commissioner-",
                    "Admitted-Commissioner-",
                    "NonAdmitted-Commissioner-",
                    "New-Periods-Commissioner-"
    ]

    # Set parameters for consultant rtt looper.
    parameters = [years, keywords, browser] 
    
    # Extract the data.
    consultant_rtt_dictionary, total_days_counter = consultant_rtt_looper(parameters)
    
    # Set start and end indices.
    start_day_index = (start_year - collective_start_year)*365
    end_day_index = start_day_index + total_days_counter

    # Collect data back into dictionary.
    consultant_rtt_dictionary = reindex_data(start_day_index, end_day_index, no_days, consultant_rtt_dictionary)  
    
    return consultant_rtt_dictionary


def mortality_domain_specifier() -> str:
    """ Specify domains for the mortality data.

    This function sets the domain for the mortality data for a given year period.
 
    Returns
    ----------
        domain : str
            Domain for the dataset.     
    """    
    # Set domain.
    domain = "https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/the21stcenturymortalityfilesdeathsdataset"

    return domain
    

def mortality_extractor(            
                            http_request: Request,
                            sheet_name: str,
                            mortality_dictionary: dict
    ) -> dict:
    """ Extracts mortality data for a given year.
    
    This function extracts the relevant data from the mortality dataset for
    a given month.
     
    Parameters
    ----------                                  
        http_request: str
            String for the http request to access the data.                                
        sheet_name: str
            String indicating the year to extract data from..
        mortality_dictionary: str
            Dictionary to store the data.                                                                                               
            
    Returns
    ----------
        mortality_dictionary: dict
            Dictionary to store the data.   
    """

    # Extract the data. 
    dataset = pd.read_excel(urlopen(http_request).read(),
                            header=1,
                            sheet_name=sheet_name)
    
    # Initialise variable to count the number of people that died
    # in the year with a given ICD-10 code.
    N_IC = 0

    # Extract the ICD codes.
    ICD_label = [x for x in list(dataset.columns) if "ICD" in x][0]

    # Loop through the ICD-10 cardiology codes.
    for code in ICD_10_cv:
        
        # Accumulate the number of deaths for each code.
        N_IC += dataset["NDTHS"][dataset[ICD_label] == code].sum()
        
    # Interpolate the data.
    mortality_dictionary["N_IC_processed"].extend([N_IC/365 for _ in range(365)])
    mortality_dictionary["N_IC_raw"].append(N_IC)

    return mortality_dictionary


def mortality_looper(parameters: list):
    """ Extracts mortality data for all requested years.

    This function extracts the relevant data from the mortality dataaset for all
    requested years.
     
    Parameters
    ----------   
        parameters: list
            List of relevant parameters.
    
    Returns
    ----------
        consultant_rtt_dictionary: dict 
            Dictionary to store the data.   
        total_days_counter: int
            Counter for the total number of days the data has been extracted for.
    """
    # Unpack parameters.
    years = parameters[0]
    keywords = parameters[1]
    browser = parameters[2]
    
    # Initialise temporary dictionary.
    mortality_dictionary = add_data_versions({ "N_IC": []})

    # Specify the domain.     
    domain = mortality_domain_specifier()
    
    # Access the domain. 
    browser.get(domain)

    # Initialise dataset links list.    
    dataset_links = []
 
    # Extract the dataset links and add them to the list. 
    dataset_links.extend(extract_domain_links(browser, keywords, ".xls"))

    # Set up http request.
    http_request = Request(dataset_links[0], headers={'User-Agent': 'Mozilla/5.0'})
    
    # Initialise time points for the data interpolation.
    t_points = []
    
    # Initialise total days counter. 
    total_days_counter = 0

    # Initialise leap year tracker.
    leap_year_tracker = 0
    
    # Loop through all the years.
    for year in years:

        # Set the sheet name.
        if year < 10:
            sheet_name = "200" + str(year)
        else:
            sheet_name = "20" + str(year)

        try:

            # Extract the data. 
            mortality_dictionary = mortality_extractor(http_request, sheet_name, mortality_dictionary)
      
            if leap_year_tracker%4==0:
                
                # Incremenet the number of days counter         
                total_days_counter += 366
    
            else:

                # Incremenet the number of days counter         
                total_days_counter += 365
             
            # Add a time point at the beginning of the month.
            t_points.append(total_days_counter)
            
        except:
            pass
        
        leap_year_tracker += 1
        
    # Set the ignore data labels list as we don't need to interpolate everything.
    ignore_data_labels = ["N_IC_processed"]

    # Interpolate the data.
    mortality_dictionary = interpolate_data_dictionary(
                                                    mortality_dictionary,
                                                    t_points,
                                                    total_days_counter,
                                                    ignore_data_labels
    )
    
    return mortality_dictionary, total_days_counter


def extract_mortality_data(
                                browser: selenium.webdriver.chrome.webdriver.WebDriver,
                                collective_start_year: int,
                                collective_end_year: int,
                                no_days: int, 
    ) -> dict:
    """ Manages mortality data extraction.
    
    This function manages the extraction of the mortality data.
    
     
    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        no_days: int
            Total number of days to extract data for. 
 
    Returns
    ----------
        mortality_dictionary: dict
            Dictionary to store the data.   
    """
    # Set start year for diagnostics data.
    start_year = 1
    
    # Set years to search for data.    
    years = list(range(start_year, collective_end_year+1))

    # Set keywords for link extraction.
    keywords = ["21stcenturymortality20"]

    # Set parameters for mortality looper.
    parameters = [years, keywords, browser] 

    # Extract the data.
    mortality_dictionary, total_days_counter = mortality_looper(parameters)
    
    # Set start and end indices.
    start_day_index = (start_year - collective_start_year)*365
    end_day_index = start_day_index + total_days_counter

    # Reindex the extracted data. 
    mortality_dictionary = reindex_data(start_day_index, end_day_index, no_days, mortality_dictionary)
    
    return mortality_dictionary


def extract_all_data(
                        browser: selenium.webdriver.chrome.webdriver.WebDriver,
                        collective_start_year: int,
                        collective_end_year: int,
                        collective_end_month: int, 
                        max_delay:int
    ):
    """ Extracts all data.
    
    This function extracts all the data for requested years.
     
    Parameters
    ----------           
        browser: selenium.webdriver.chrome.webdriver.WebDriver
            Selenium browser for extracting data from websites.
        collective_start_year: int
            Year to try and start extracting data from.
        collective_end_year: int
            Year to stop extracting data from.
        collective_end_month: int
            Month to stop extracting data from.
        max_delay: int
            Upper bound on the delay between booking and having a GP appointment.
    
    Returns
    ----------
        data_dictionary: Dict 
            Dictionary to store the data.   
    """
    
    # Compute the number of days to extract data for.
    collective_start_date = datetime.date(collective_start_year, 1, 1)
    collective_end_date = datetime.date(collective_end_year, collective_end_month, 1)

    no_days = (collective_end_date - collective_start_date).days

    # Extract primary care data.
    # Extract mortality data.
    mortality_data = extract_mortality_data( 
                                            browser, 
                                            collective_start_year,
                                            collective_end_year,
                                            no_days
    )    

    print("Mortality done")
    
    primary_care_data = extract_primary_care_data( 
                                                    browser, 
                                                    collective_start_year,
                                                    collective_end_year,
                                                    no_days,
                                                    max_delay
    )
    print("Primary care done")


    # Extract qof data.
    qof_data = extract_qof_data(
                                    browser, 
                                    collective_start_year,
                                    collective_end_year,
                                    no_days
    )
    print("QOF done")



    # Extract consultant rtt data.
    consultant_rtt_data = extract_consultant_rtt_data(
                                            browser, 
                                            collective_start_year,
                                            collective_end_year,
                                            no_days
    )
    
    print("Consultant rtt done")

    # Extract hes data.
    hes_data = extract_hes_data( 
                                    browser, 
                                    collective_start_year,
                                    collective_end_year,
                                    no_days
    )           

    # Extract diagnostics data.
    diagnostics_data = extract_diagnostics_data(
                                                    browser, 
                                                    collective_start_year,
                                                    collective_end_year,
                                                    no_days
    )                                                         

    print("Diagnostics done")
    
    # Extract stats wales data ( this is hard coded because we only use it once to compute 
    # beta)
    stats_wales_data = {"N_L_W_raw": [3211292, 3220310, 3229700, 3236410,
                                np.mean([3242338, 3233742, 3233879]),
                                np.mean([3237764, 3236819, 3240583])]}

    print("Stats wales done")


    # Collect all data together in one dictionary.
    data_dictionary = {
                        **stats_wales_data,
                        **mortality_data,
                        **consultant_rtt_data,
                        **hes_data,
                        **diagnostics_data,
                        **primary_care_data,
                        **qof_data
    }
    
    # Save data dictionary.    
    np.save("data_dictionary.npy", data_dictionary, allow_pickle=True)



"""======================================================================================================================"""
"""======================================================================================================================"""
###  Plotting functions
"""======================================================================================================================"""
"""======================================================================================================================"""

def generate_time_periods(
                            dates: dict,
                            no_pre_lockdown_days: int,
                            no_post_lockdown_days: int,
                            no_projection_days: int
    ):
    """ Generates the indices for all the time periods. 
    
    This function generates the dates for the pre-lockdown, lockdown,
    post-lockdown and projection time periods. The dates are then used to 
    compute the indices of these periods within the entire simulation. 
    
    Parameters
    ----------           
        dates: dict
            Dictionary containing key dates.
        no_pre_lockdown_days: int
            The number of days defining the pre-lockdown period.
        no_post_lockdown_days: int
            The number of days defining the post-lockdown period.
        no_projection_days: int
            The number of days defining the projection period.            
    Returns
    ----------
        time_periods: dict
            Dictionary containing the indices of the time periods. 
        dates: dict
            Dictionary containing key dates.
    """
    # Unpack the key dates dictionary.
    lockdown_start_date = dates["lockdown start date"]
    lockdown_end_date = dates["lockdown end date"]

    # Compute the number of lockdown days.
    no_lockdown_days = (lockdown_end_date - lockdown_start_date).days

    # Compute pre-lockdown period start and end dates.
    pre_lockdown_start_date = lockdown_start_date - timedelta(no_pre_lockdown_days)  
    pre_lockdown_end_date = lockdown_start_date

    # Compute post-lockdown period start and end dates.
    post_lockdown_start_date = lockdown_end_date
    post_lockdown_end_date = lockdown_end_date + timedelta(no_post_lockdown_days)  

    # Compute total number of simulation days.
    no_simulation_days = no_pre_lockdown_days + no_lockdown_days\
                       + no_post_lockdown_days + no_projection_days

    # Compute projection period start and end dates.
    projection_start_date = post_lockdown_end_date
    projection_end_date = post_lockdown_end_date + timedelta(no_projection_days)  

    # Compute simulation start and end dates.
    simulation_start_date = projection_end_date - timedelta(no_simulation_days)
    simulation_end_date = projection_end_date 

    # Update dates dictionary.
    dates["pre lockdown start date"] = pre_lockdown_start_date
    dates["pre lockdown end date"] = pre_lockdown_end_date
    dates["post lockdown start date"] = post_lockdown_start_date
    dates["post lockdown end date"] = post_lockdown_end_date
    dates["projection start date"] = projection_start_date
    dates["projection end date"] = projection_end_date
    dates["simulation start date"] = simulation_start_date
    dates["simulation end date"] = simulation_end_date
    dates["no projection days"] = no_projection_days
    dates["projection dates"] = [projection_start_date + timedelta(days=x) for x in range(no_projection_days+1)]
    dates["simulation dates"] = [simulation_start_date + timedelta(days=x) for x in range(no_simulation_days+1)]

    # Compute the indices for each period.
    pre_lockdown_start_index = (dates["pre lockdown start date"]-dates["start date"]).days
    pre_lockdown_end_index = (dates["pre lockdown end date"]-dates["start date"]).days

    lockdown_start_index =  (dates["lockdown start date"]-dates["start date"]).days
    lockdown_end_index = (dates["lockdown end date"]-dates["start date"]).days

    post_lockdown_start_index = (dates["post lockdown start date"]-dates["start date"]).days
    post_lockdown_end_index = (dates["post lockdown end date"]-dates["start date"]).days

    projection_start_index = (dates["projection start date"]-dates["start date"]).days
    projection_end_index = (dates["projection end date"]-dates["start date"]).days
    time_periods = {
                            "pre lockdown": [pre_lockdown_start_index, pre_lockdown_end_index],
                            "lockdown": [lockdown_start_index, lockdown_end_index],
                            "post lockdown": [post_lockdown_start_index, post_lockdown_end_index],
                            "projection": [projection_start_index, projection_end_index]
    }   

    return time_periods, dates


def plot_all_data(data_plotting_parameters: dict):
    """ Plots all extracted data.
    
    This function plots all the extracted data.
    
    Parameters
    ----------           
        data_plotting_parameters: dict
            Dictionary containing parameters for plotting.
    """


    # Loop through all the data in the data dictionary.
    for data_label in data_plotting_parameters["data dictionary"].keys():
        
        if data_plotting_parameters["data type"] in data_label:
            
            print("Plotting ", data_label)
            plot_data(data_plotting_parameters, data_label)
        
            
def determine_non_zero(
                        data,
                        dates,
                        data_type,
                        tick_date_formatting
    ):
    """ Function which determines the dates for which we have data for a given dataset. 

    Data is available for different datasets over different perios. This function determines
    the dates (and the data) for which data has been extracted for a given dataset. 
    
    Parameters
    ----------           
        data: np.array
            Array of data. 
        dates: dict
            Dictionary containing the key dates i.e. start date. 
        data_type: str
            String indicating whether this is raw or processed data. 
        tick_date_formatting: str
            String indicating the formatting used for the ticks representing the date. 
            
    Returns
    ----------           
        non_zero_dates: list
            List containing the dates for whcih data was extracted.
        non_zero_data: np.array
            Array containing the data. 
    """
    # Compute the nonzero indices.
    non_zero_indices = np.nonzero(data)
  
    # Compute the start and end indices of the nonzero part of the data array.
    data_start_index = np.min(non_zero_indices)
    data_end_index = np.max(non_zero_indices)
    non_zero_data = data[non_zero_indices]
    
    # Set the data start date.
    data_start_date = dates["start date"] + timedelta(days = int(data_start_index))
    
    # Compute the number of days that we have data available within the fitting period.
    no_days = data_end_index - data_start_index
    
    # Compute the dates that we have data available within the fitting period.
    non_zero_dates = [data_start_date + timedelta(days=x) for x in range(no_days+1)]

    if data_type=="raw":
    
        adjusted_non_zero_indices = non_zero_indices[0] - data_start_index
        non_zero_dates = [non_zero_dates[x] + timedelta(1) for x in adjusted_non_zero_indices]
    
    else:
        if tick_date_formatting=="%Y":
            adjusted_non_zero_indices = non_zero_indices[0] - data_start_index
            non_zero_dates = [non_zero_dates[x] + timedelta(20) for x in adjusted_non_zero_indices]    
      
    return non_zero_dates, non_zero_data


def plot_data(data_plotting_parameters: dict, data_label: str):               
    """ Plots extracted data.
    
    This function plots some extracted data.
    
    Parameters
    ----------           
        data_plotting_parameters: dict
            Dictionary containing parameters for plotting.
        data_type: str
            String indicating whether the user wants to plot the simulation 
            or plotting data. 
    """
    # Unpack plotting parameters.
    data_label = data_label
    data = data_plotting_parameters["data dictionary"][data_label]
    data_type = data_plotting_parameters["data type"]
    dates = data_plotting_parameters["dates"]
    save_flag = data_plotting_parameters["save flag"]
    show_flag = data_plotting_parameters["show flag"]
    text_fontsize = data_plotting_parameters["text fontsize"]
    ticks_fontsize = data_plotting_parameters["ticks fontsize"]

    # Initialise plot.
    fig, axes = plt.subplots()

    data_label = data_label.replace("_" + data_type, "")
    
    # Compute the order of magnitude for this data. 
    order_magnitude = np.floor(np.log10(np.nanmax(data)))

    simulation_data = data_plotting_parameters["data dictionary"][data_label + "_processed"]

    raw_data = data_plotting_parameters["data dictionary"][data_label + "_raw"]

    # Set appropriate tick labels.     
    if data_plotting_names[data_label][2]==annual_width:
        tick_date_formatting = "%Y"
        axes.set_xlabel("Year", fontsize=text_fontsize)

    else:
        tick_date_formatting = "%m-%Y"
        axes.set_xlabel("Date", fontsize=text_fontsize)

    # Determine the non zero data and the corresponding dates. 
    non_zero_simulation_dates, non_zero_simulation_data  = determine_non_zero(simulation_data, dates, "processed", tick_date_formatting)
    non_zero_raw_dates, non_zero_raw_data = determine_non_zero(raw_data, dates, "raw", tick_date_formatting)

    # If plotting a rate then multiply by 100.
    if "absence_rate" in data_label or "p_conds" in data_label:
    
        non_zero_raw_data = non_zero_raw_data*100
        non_zero_simulation_data = non_zero_simulation_data*100

        
    # Determine which colour to set the lockdown lines.
    period_colour_dictionary = {
                                    "lockdown": "k",
    }

    # Partition the dates dictionary to get labelled dictionaries for plotting.
    period_labels = generate_period_labels(dates, True)

    # Plot data.
    if data_type=="processed":
        
        # Rescale the data so that we get ticks in scientific notation with 10^order of magnitude 
        # plotted at the top of the y-axis. 
        non_zero_simulation_data = non_zero_simulation_data/np.power(10, order_magnitude)
        
        axes.plot(non_zero_simulation_dates, non_zero_simulation_data, color="r", linewidth=3)

        no_days_available = (non_zero_simulation_dates[-1] -  non_zero_simulation_dates[0]).days
                
        # Plot the ticks.
        plot_ticks(
                    ticks_fontsize,
                    tick_date_formatting,
                    non_zero_simulation_dates,
                    order_magnitude,                
                    no_days_available
        )
        
    else:
        
        # Rescale the data so that we get ticks in scientific notation with 10^order of magnitude 
        # plotted at the top of the y-axis. 
        non_zero_raw_data = non_zero_raw_data/np.power(10, order_magnitude)

        axes.bar(
                    non_zero_raw_dates,
                    non_zero_raw_data,
                    width=data_plotting_names[data_label][2],
                    color="r",
                    alpha=1,
                    edgecolor = "black"
        )
    
        no_days_available = (non_zero_raw_dates[-1] -  non_zero_raw_dates[0]).days

        # Plot the ticks.
        plot_ticks(
                    ticks_fontsize,
                    tick_date_formatting,
                    non_zero_raw_dates,
                    order_magnitude,                
                    no_days_available
        )

    # Add lines to indicate lockdown period.
    axes, lines, labels = add_intervention_lines(
                                        period_labels,
                                        period_colour_dictionary,
                                        axes
    )

    # Set title and axis labels. 
    axes.set_ylabel(data_plotting_names[data_label][1], fontsize=text_fontsize)

    # Compute figure name for saving purposes.
    if data_type=="processed":

        try:
            os.mkdir(os.getcwd() + "/figures/data/simulation/")
        except:
            pass

        figure_name = os.getcwd() + "/figures/data/simulation/" + data_label
    else:

            
        try:
            os.mkdir(os.getcwd() + "/figures/data/raw/")
        except:
            pass

        figure_name = os.getcwd() + "/figures/data/raw/" + data_label
        
    # Format the figure and save or show as requested.
    format_figure(fig, lines, labels, figure_name, save_flag, show_flag)
    
    plt.close("all")


def plot_ticks(
                ticks_fontsize: int,
                tick_date_formatting:str,
                dates: dict,
                order_magnitude: float,
                no_days_available: int,
    ):
    """ Plots the ticks.
    
    This function plots the ticks for the axes of the plot.
    
    Parameters
    ----------           
        ticks_fontsize : int
            Ticks fontsize.
        day_interval : int
            Interval measured in days for the x-axis ticks.
        dates : dict
            Dictionary containing key dates.      
    """

    # Plot the ticks.
    plt.xticks(ticks=dates, fontsize=ticks_fontsize)

    plt.yticks(fontsize=ticks_fontsize)
  
    # Format the y-axis ticks.
    plt.ticklabel_format(
                            axis="y",
                            style="plain",
                            scilimits=(0,0),
                            useMathText=True
    )
    
    # Set the exponent label.
    exponent_label = r'{\times}10^{%s}' % (int(order_magnitude))
    
    # Place the exponent label at the top of the y-axis.
    plt.text(0.0, 1.01, "${}$".format(exponent_label), fontsize=ticks_fontsize, transform = plt.gca().transAxes)
        
    # Format the x-axis ticks.
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(tick_date_formatting))    

    # Standardise the number of x-axis ticks.
    if tick_date_formatting=="%Y":

        if no_days_available/365 > 8:
            date_tick_interval = 730
        else:
            date_tick_interval = 366

    else:
    
        date_tick_interval = int(no_days_available/5) 

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_tick_interval))
         
    plt.gca().xaxis.set_tick_params(rotation = 30) 

    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(8))


def generate_period_labels(dates: dict, lockdown_only=False):
    """ Generates period labels.
    
    This function generates dictionaries for the lockdown and 
    projection periods which contain the start date, end date 
    and an appropriate label.
        
    Parameters
    ----------
        dates : dict
            Dictionary containing key dates.           
        lockdown_only : bool
            Boolean indicating whether the user has requested
            only the lockdown period to be labelled.
        
    Returns
    ----------
        labels : list
            List of dictionaries with labels for given periods.
    """
    # Generate lockdown period label dictionary.
    lockdown_label =  {
                        "start_date": dates["lockdown start date"],
                        "end_date": dates["lockdown end date"],
                        "label": "lockdown"
    }

    # Generate perdiction period label dictionary.
    projection_label =  {
                        "start_date": dates["projection start date"],
                        "end_date": dates["projection end date"],
                        "label": "projection period"
    }
    
     
    # Return data as requested.
    if lockdown_only:
        
        labels = [lockdown_label]
    
    else:
        
        labels = [lockdown_label, projection_label]
        
    return labels


def add_intervention_lines(
        intervention_rule_sets: list,
        intervention_colour_dictionary: dict,
        axes: plt.axes
    ) -> list:
    """ Add intervention lines to the plot.

    This function adds the intervention lines to the plot. These could be 
    the lockdown lines and any interventions like increases in capacity. 
    By lines, we mean lines on the plot that indicate the time interval over
    which the intervention applies.
    
    
    Parameters
    ----------
        intervention_rule_sets : list
            List containing the intervention rule sets dictionaries.
        intervention_colour_dictionary : list
            List containing the intervention colour dictionaries. These specify
            what colour to use for the intervention lines.
        axes: plt.axes,
            Axes with data plotted.
  
    Returns
    ----------
        plot_data : list
            List containing the axes, lines and labels.

    """ 
    # Initialise empty lists.
    lines = []
    labels = []
            
    # Loop through the intervention rule sets.
    for rule_set in intervention_rule_sets:
            
        # Add the lines to the plot with the metadata specified in the intervention 
        # rule sets and colour dictionary.
        lines.append(
                        axes.axvline(
                                        pd.to_datetime(rule_set["start_date"]),
                                        color=intervention_colour_dictionary[rule_set["label"]],
                                        lw=3,
                                        linestyle="-",
                                        alpha=1,
                                        label=rule_set["label"]
                                    )
                    )
        # Add the lines to the plot with the metadata specified in the intervention 
        # rule sets and colour dictionary.
        lines.append(
                        axes.axvline(
                                        pd.to_datetime(rule_set["end_date"]),
                                        color=intervention_colour_dictionary[rule_set["label"]],
                                        lw=3,
                                        linestyle="-",
                                        alpha=1,
                                        label=rule_set["label"]                                       
                                    )
                    )
        # Add the intervention label.
        labels.append(rule_set["label"])
    
    # Store data in a list and return.
    plot_data = [axes, lines, labels]

    return plot_data


def format_figure(
        fig: figure.Figure,
        lines: list,
        labels: list,
        figure_name: str,
        save_flag: bool,
        show_flag: bool
    ) -> None:
    """ Formats the figures.

    Helper function for reformatting the figures for display purposes.

    Parameters
    ----------
        fig : fig
            Figure of plot to save.
        lines : plt.lines
            Intervention lines of the plot.
        labels : plt.labels
            Labels of the plot.
        figure_name : str
            Name to save the figure with.
        save_flag : bool
            Flag indicating whether or not to save the figure.
        show_flag : bool
            Flag indicating whether or not to show the figure. 
    """ 
    # Format the dates.
    fig.autofmt_xdate()
    
    # Delete spare copies of the intervention lines.
    del lines[1::2]
    
    # Add legend.
    fig.legend(lines,  labels, loc="lower center", ncol=3, prop={"size":20})
    
    # Resize the figure.
    fig = resize_figure(fig)
    
    # Save the figure if requested.
    if save_flag:
        
        # Save figure.
        fig.savefig(figure_name + ".png", dpi=plt.gcf().dpi) 
        
    # Show the figure if requested.
    if show_flag:
        
        # Show figure.
        plt.show()


def resize_figure(fig: figure.Figure):
    """ Resize figure.
    
    This function resizes the figure for plotting and saving purposes.  
    
    Parameters
    ----------
        fig : fig
            Figure of plot to save.
    
    Returns
    ----------
        fig : fig
            Figure of plot to save.
    """ 
    # Get figure manager.
    manager = plt.get_current_fig_manager()
    
    # Resize figure.
    manager.resize(*manager.window.maxsize())

    fig = plt.gcf()
    
    fig.set_size_inches(    
                            manager.window.maxsize()[0]/plt.gcf().dpi,
                            manager.window.maxsize()[1]/plt.gcf().dpi,
                            forward=False
                        )

    return fig


def compute_data_statistics(
                                data: np.array,
                                start_index: int,
                                end_index: int
    ):
    """ Computes data statistics.

    This function computes relevant statistics of the dataset. For now it's
    just the mean. 
    
    
    Parameters
    ----------
        data: np.array
            Array containing the data.
        start_index: int
            Index of first data point.
        end_index: int
            Index of last data point.

    Returns
    ----------
        data_stats: dict
            Dictionary containing the data statistics. 
       
    """ 
    # Find data within the period.
    period_data = data[start_index:end_index]
    
    # Find non zero data.
    non_zero_data = period_data[period_data!=0]

    # Compute statistics.
    data_stats = {
                    "mean": np.nanmean(non_zero_data),   
    }
    
    return data_stats


def compute_all_data_statistics(
                                data_dictionary: dict,
                                time_periods: dict,
    ):
    """ Computes data statistics.

    This function computes relevant statistics of the dataset. For now it's
    just the mean. 
    
    
    Parameters
    ----------
        data: np.array
            Array containing the data.
        start_index: int
            Index of first data point.
        end_index: int
            Index of last data point.

    Returns
    ----------
        data_stats: dict
            Dictionary containing the data statistics. 
       
    """ 

    # Extract simulation data. 
    data_dictionary = extract_processed_data(data_dictionary)

    # Initialise a dictionary for storing statistics for a given period.
    period_data_statistics = dict.fromkeys(data_dictionary.keys(), 0)

    # Initialise a dictionary for storing dictionaries for all periods.
    data_statistics = {k: copy.deepcopy(period_data_statistics) for k in time_periods.keys()}

    # Loop through the time period dictionary.
    for time_period_label in time_periods.keys():

        # Obtain the start and end indices of this time period.
        start_index = time_periods[time_period_label][0]
        end_index = time_periods[time_period_label][1]
    
        for data_key, data in data_dictionary.items():
            
            # Compute statistics.
            try:                
                data_stats = compute_data_statistics(            
                                                        data,
                                                        start_index,
                                                        end_index
                )
            except:
                print(data_key)
            
            # Store statistics.
            data_statistics[time_period_label][data_key] = data_stats["mean"]

    return data_statistics


def extract_processed_data(data_dictionary:dict):
    """ Extracts the processed data from the data dictionary.

    This function creates a new data dictionary with just the processed data. 
    
    
    Parameters
    ----------
        data_dictionary: dict
            Dictionary of data with processed and raw entries. 
    
    Returns
    ----------
        data_dictionary: dict
            Dictionary of data with processed entries. 
       
    """ 
    # Extract the processed data from the dictionary dictionary. 
    processed_data = {k.replace("_processed", ""):data_dictionary[k] for k in list(data_dictionary.keys()) if "_processed" in k}

    return processed_data


 
def draw_sample(lb: float, ub: float) -> float:
    """ Draws sample from uniform distribution.

    This function draws a sample from a uniform distribution, used for the 
    Monte Carlo simulation.
    
    
    Parameters
    ----------
        lb: float 
            Lower bound of distribution.
        ub: float 
            Upper bound of distribution.
        
    Returns
    ----------
        sampled_data: float
            Sample from uniform distribution [lb, ub]

       
    """      
    # Draw sample and return.
    sampled_data = np.random.uniform(
                                       lb,
                                       ub
    )
        
    return sampled_data


#### TODO - Check this works
def compute_beta(data_dictionary: dict):
    """ Computes the value of beta. 

    This function computes the value of beta from the data on the number 
    of people registered at GP practices in England and Wales. We use estimation 
    method 3 from the paper.
    
    
    Parameters
    ----------
        data_dictionary: dict
            Dictionary of data.
        
    Returns
    ----------
        beta: float 
            Ratio of people registered at a GP practice in Wales v England.
       
    """
    # Extract necessary data
    N_L_E = data_dictionary["N_L_E_raw"]
    N_L_W = data_dictionary["N_L_W_raw"]


    # Extract non-zero data
    non_zero_N_L_E = N_L_E[N_L_E!=0][-len(N_L_W):]

    beta = N_L_W/non_zero_N_L_E
    
    return beta[0]


def create_parameters_time_series(
                                    optimisation_results: dict,
                                    no_days_dictionary: dict,
                                    intervention_dictionary: dict
    ):
    """ Create time series for each parameter.
    
    Parameters
    ----------
        no_days_dictionary: dict
            Dictionary containing the number of days in each epoch.
        intervention_dictionary: dict
            Dictionary containing the interventions.
        optimisation_results: dict
            Dictionary containing the optimisation results.
    Returns
    ----------
       model_output: dict
            Dictionary of model outputs for each stock.
    
    """
    # Extract results from each epoch.
    pre_lockdown_results = optimisation_results["pre lockdown"]
    lockdown_results = optimisation_results["lockdown"]
    post_lockdown_results = optimisation_results["post lockdown"]
    
    # Initialise time series.
    parameters_time_series = {k: [] for k in pre_lockdown_results["parameters time series"].keys()}

    # Create time series from parameter values.
    for key, value in pre_lockdown_results["parameters time series"].items():
        parameters_time_series[key] = pd.Series(index=range(no_days_dictionary["pre lockdown"]), data=value)
      
    for key, value in lockdown_results["parameters time series"].items():
        parameters_time_series[key] = pd.concat(
                                    [
                                        parameters_time_series[key],    
                                        pd.Series(index=range(no_days_dictionary["lockdown"]), data=value)
                                    ],
                                    ignore_index=True
        )
       
    for key, value in post_lockdown_results["parameters time series"].items():
        parameters_time_series[key] = pd.concat(
                                    [
                                        parameters_time_series[key],    
                                        pd.Series(index=range(no_days_dictionary["post lockdown"]), data=value)
                                    ],
                                    ignore_index=True
        )
      
    # Perform Monte Carlo sampling on the free parameters in the post lockdown epoch.     
    samples = perform_MC_sampling(post_lockdown_results["parameters values"])
           
    for key, value in post_lockdown_results["parameters values"].items():


        if key in samples.keys():
            
            parameter_value = samples[key]
                      
        else:
                
            parameter_value = value

        # Update parameter value if an intervention for this parameter has 
        # been specified in the parameter dictionary.
        if intervention_dictionary[key]["flag"]:
            
            parameter_value = parameter_value*intervention_dictionary[key]["multiplier"]

            
        if key!="beta_CC":
            
            parameters_time_series[parameters_plotting_names[key][0]]= pd.concat(
                                            [
                                                parameters_time_series[parameters_plotting_names[key][0]],    
                                                pd.Series(index=range(no_days_dictionary["projection"]), data=parameter_value)
                                            ],
                                            ignore_index=True
            ) 
        
    return parameters_time_series
        
 
def perform_MC_sampling(parameters_values: dict):
    """ Performs Monte Carlo sampling. 

    This function performs the Monte Carlo sampling procedure outlined in 
    the paper. 

    Parameters
    ----------
        parameters_values: dict
            Dictionary containing the value of each parameter.
    
    Returns
    ----------
       samples: dict
            Dictionary containing the sampled parameter values. 
    """
    # Sample beta_CC and compute the induced change in the other
    # consultant proportion parameters.  
  
    beta_CC = parameters_values["beta_CC"]
    beta_CT = parameters_values["beta_CT"]
    beta_CD = parameters_values["beta_CD"]
    beta_CG = parameters_values["beta_CG"]

    beta_CC_sample = draw_sample(0.95*beta_CC, 1.05*beta_CC)
    
    sample_difference =  beta_CC_sample - beta_CC
    
    gamma = sample_difference/(beta_CT + beta_CD + beta_CG)
    
    beta_CT_sample = (1-gamma)*beta_CT
    beta_CD_sample = (1-gamma)*beta_CD
    beta_CG_sample = (1-gamma)*beta_CG 
  
    samples = {
                "beta_CC": beta_CC_sample,
                "beta_CT": beta_CT_sample, 
                "beta_CD": beta_CD_sample,
                "beta_CG": beta_CG_sample
    }
    

    return samples 


def initialise_model_outputs(no_simulation_runs: int, no_time_steps: int):
    """ Initialise model output dictionary.

    Parameters
    ----------
        no_simulation_runs: int 
            The number of simulation runs.
        no_time_steps: int
            The number of time steps in each simulation.
    
    Returns
    ----------
        model_outputs: dict
            Dictionary containing the model outputs for each stock.
       
    """
    # Initialise the model outputs.
    model_outputs = {
                        "Symptomatic population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Died in symptomatic population": np.zeros((no_simulation_runs, no_time_steps)),
                        "GP waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Died on GP waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Diagnostic waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Died on diagnostic waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Consultant waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Died on consultant waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Treatment waiting list population": np.zeros((no_simulation_runs, no_time_steps)),
                        "Died on treatment waiting list population": np.zeros((no_simulation_runs, no_time_steps))
    }

    return model_outputs


def set_initial_conditions(stocks: dict):
    """ Sets the initial conditions. 

    This function sets the initial conditions into the correct format 
    for the PYSD model. 

    Parameters
    ----------
        stocks: dict 
            Dictionary containing the initial conditions for the stocks.
        
    Returns
    ----------
        initial_conditions: dict 
            Dictionary containing the initial conditions for the stocks
            with the right labels for the PYSD model.
       
    """
    # Set the initial conditions.
    initial_conditions = {
                            "Symptomatic population": stocks["P_S"],
                            "GP waiting list population": stocks["P_G"],
                            "Diagnostic waiting list population": stocks["P_D"],
                            "Consultant waiting list population": stocks["P_C"],
                            "Treatment waiting list population": stocks["P_T"], 
                            "Died on GP waiting list population": 0,
                            "Died on diagnostic waiting list population": 0,
                            "Died on consultant waiting list population": 0,
                            "Died in symptomatic population": 0,
                            "Died on treatment waiting list population": 0
    }
    
    return initial_conditions


def set_parameter_time_series(parameters: dict):
    """ Sets the parameter time series. 

    This function sets the parameter time series into the correct format 
    for the PYSD model. 

    Parameters
    ----------
        parameters: dict 
            Dictionary containing the parameter time series.
        
    Returns
    ----------
        parameter_time_series: dict 
            Dictionary containing the parameter time series
            with the right labels for the PYSD model.
       
    """
    # Set the simulation parameters..
    parameter_time_series = {
                    "GP appointment supply": parameters["alpha_G"],
                    "Diagnostic appointment supply": parameters["alpha_D"],
                    "Consultant appointment supply": parameters["alpha_C"],
                    "Treatment appointment supply": parameters["alpha_T"],
                    "Proportion of GP appointments with discharge to symptomatic population": parameters["beta_GS"],
                    "Proportion of diagnostic appointments with discharge to GP":  parameters["beta_DG"],
                    "Proportion of diagnostic appointments with consultant referral":  parameters["beta_DC"],
                    "Proportion of consultant appointments with discharge to GP":  parameters["beta_CG"],
                    "Proportion of consultant appointments with treatment referral":  parameters["beta_CT"],
                    "Proportion of consultant appointments with diagnostic referral":  parameters["beta_CD"],
                    "Proportion of GP appointments with consultant referral":  parameters["beta_GC"],
                    "Proportion of GP appointments with diagnostic referral": parameters["beta_GD"],
                    "Probability of dying with symptoms": parameters["beta_M"],
                    "Proportion of symptomatic patients booking GP appointments": parameters["beta_SG"],
                    "Symptom development rate": parameters["beta_S"]
    }

    return parameter_time_series


def update_model_outputs(model_outputs: dict, model_output: dict, index: int):
    """ Updates the model output dictionary.

    This function updates the model output dictionary after each simulation run.

    Parameters
    ----------
        model_outputs: dict 
            Dictionary containing model outputs for each stock for all simulation
            runs.
        model_output: int
            Dictionary containing the model output from this simulation run.
        index: int
            Index of simulation run. 
    
    Returns
    ----------
        model_outputs: dict
            Dictionary containing model outputs for each stock for all simulation
            runs.
       
    """
    # Update model outputs.
    model_outputs["Symptomatic population"][index, :] = model_output["Symptomatic population"]
    model_outputs["Died in symptomatic population"][index, :] = model_output["Died in symptomatic population"]
    model_outputs["GP waiting list population"][index, :] = model_output["GP waiting list population"]
    model_outputs["Died on GP waiting list population"][index, :] = model_output["Died on GP waiting list population"]
    model_outputs["Diagnostic waiting list population"][index, :] = model_output["Diagnostic waiting list population"]
    model_outputs["Died on diagnostic waiting list population"][index, :] = model_output["Died on diagnostic waiting list population"]
    model_outputs["Consultant waiting list population"][index, :] = model_output["Consultant waiting list population"]
    model_outputs["Died on consultant waiting list population"][index, :] = model_output["Died on consultant waiting list population"]
    model_outputs["Treatment waiting list population"][index, :] = model_output["Treatment waiting list population"]
    model_outputs["Died on treatment waiting list population"][index, :] = model_output["Died on treatment waiting list population"]

    return model_outputs


def run_monte_carlo_simulation(simulation_parameters: dict):
    """ Run Monte Carlo simulation.
    
    Parameters
    ----------
        simulation_parameters: dict
            Dictionary containing simulation parameters.

    Returns
    ----------
       model_outputs: dict
            Dictionary of model outputs for each stock.
    
    """
    # Extract the simulation parameters. 
    no_simulation_runs = simulation_parameters["no simulation runs"]
    no_time_steps = simulation_parameters["no time steps"]
    time_step = simulation_parameters["time step"]
    no_simulation_days = simulation_parameters["no simulation days"]
    model = simulation_parameters["model"]
    no_days_dictionary = simulation_parameters["no days dictionary"]
    intervention_dictionary = simulation_parameters["intervention dictionary"]

    # Initialise the model outputs for each stock. 
    model_outputs = initialise_model_outputs(no_simulation_runs, no_time_steps + 1)

    # Load the optimisation results.
    optimisation_results = np.load("optimisation_results.npy", allow_pickle="True").flat[0]
    
    # Set the time step
    time_step_factor = int(1/time_step)
    
    # Compute the number of time steps.
    no_time_steps =  no_simulation_days*time_step_factor
      
    # Initialise simulation run count.
    simulation_run_count = 0
    
    # Loop through simulation runs.
    for index in range(no_simulation_runs):
        
        # Run simulation.
        model_output = run_simulation(
                                    no_time_steps,
                                    time_step,
                                    model,
                                    no_days_dictionary,
                                    intervention_dictionary, 
                                    optimisation_results    
        )
        
        # Update model outputs.
        model_outputs = update_model_outputs(model_outputs, model_output, index)
            
        print("Finished simulation run : ", simulation_run_count)
        
        simulation_run_count += 1
    
    return model_outputs


def run_simulation(
                    no_time_steps,
                    time_step,
                    model,
                    no_days_dictionary,
                    intervention_dictionary,     
                    optimisation_results
    ):
    """ Run Monte Carlo simulation.
    
    Parameters
    ----------
        no_time_steps: int
            Number of time steps.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.      
        no_days_dictionary: dict
            Dictionary containing the number of days in each epoch.
        intervention_dictionary: dict
            Dictionary containing the interventions.
        optimisation_results: dict
            Dictionary containing the optimisation results.
    Returns
    ----------
       model_output: dict
            Dictionary of model outputs for each stock.
    
    """
    # Create parameter time-series.
    parameters_time_series = create_parameters_time_series(
                                                            optimisation_results,
                                                            no_days_dictionary,
                                                            intervention_dictionary
    )

    # Set the initial conditions. 
    initial_conditions = set_initial_conditions(optimisation_results["pre lockdown"]["stocks initial conditions"])

    # Run the model.
    model_output = model.run( 
                                progress=False,
                                time_step=time_step,
                                final_time=no_time_steps,
                                params=parameters_time_series,
                                initial_condition=(0, initial_conditions)
    )
    
    
    return model_output


def compute_initial_conditions(data: dict):
    """ Compute initial conditions for each stock.
    
    Parameters
    ----------
        simulation_parameters: dict
            Dictionary containing simulation parameters.

    Returns
    ----------
       initial_conditions: dict
            Dictionary of initial conditions for each stock.
    
    """
    # Compute initial conditions.
    P_S = data["p_conds"]*data["N_L_E"]
    
    P_G = data["p_conds"]*data["B_GP"]
    
    P_D = data["W_echocardiography"] + data["W_electrophysiology"]\
        + data["W_MRI_cardiology"] + data["W_CT_cardiology"] 

    P_C = data["W_incomplete_cs"] + data["W_incomplete_c"]
    
    P_T = data["W_incomplete_dta_cs"] + data["W_incomplete_dta_c"]

    initial_conditions = {
                            "P_S": P_S,
                            "P_G": P_G,
                            "P_D": P_D,
                            "P_C": P_C,
                            "P_T": P_T,
    }
    
    return initial_conditions


def direct_estimation(
                        data: dict,
                        gamma: float,
                        beta: float
    ) -> dict:
    """ Direct estimation of parameters.

    This function directly estimates some of the parameters from the processed data. 

    Parameters
    ----------
        data: dict
            Dictionary containing processed data. 
        gamma: float
            Difference in mortality rates between Wales and England. 
        beta: float 
            Ratio of people registered at a GP practice in Wales v England.
           
    Returns
    ----------
       direct_estimates: dict
            Dictionary of parameters estimated directly from data. 
    
    """
    # This is all in the paper. 
    beta_M = (1/(365*(1+beta)))*((data["N_IC"]*365)/data["N_L_E"] - gamma*beta)    

    alpha_G = data["p_conds"]*data["A_GP"]

    alpha_C = data["N_A_codes"]
    
    alpha_D = data["N_echocardiography"] + data["N_electrophysiology"] \
         + data["N_MRI_cardiology"] + data["N_CT_cardiology"]
  
    alpha_T = data["N_FCEP_codes"]
    
    beta_GC = (1/alpha_G)*(data["N_new_cs"] + data["N_new_c"] )
    
    beta_CG = data["N_FCE_codes"]/data["N_A_codes"]

    direct_estimates = {
                            "alpha_G": alpha_G,
                            "alpha_C": alpha_C,
                            "alpha_D": alpha_D,
                            "alpha_T": alpha_T,
                            "beta_GC": beta_GC,
                            "beta_CG": beta_CG,
                            "beta_M": beta_M
    }

    return direct_estimates


def direct_parameter_estimation(
                                data_statistics: dict,
                                gamma: float,
                                beta: float
    ) -> dict:
    """ Performs direct parameter estimation in each epoch.

    This function calls the direct estimation routine on the processed
    data for each epoch. 

    Parameters
    ----------
        data_statistics: dict
            Dictionary containing processed data. 
        gamma: float
            Difference in mortality rates between Wales and England. 
        beta: float 
            Ratio of people registered at a GP practice in Wales v England.
           
    Returns
    ----------
       direct_estimation_results: dict
            Dictionary of parameters estimated directly from data. 
    """
    # Estimate parameters in the pre lockdown epoch.
    pre_lockdown_estimates = direct_estimation(data_statistics["pre lockdown"], gamma, beta)

    # Estimate parameters in the lockdown epoch.
    lockdown_estimates = direct_estimation(data_statistics["lockdown"], gamma, beta)

    # Estimate parameters in the post lockdown epoch.
    post_lockdown_estimates = direct_estimation(data_statistics["post lockdown"], gamma, beta)

    # Store direct parameter estimation results.
    direct_estimation_results = {       
                                    "pre lockdown estimates": pre_lockdown_estimates,
                                    "lockdown estimates": lockdown_estimates,
                                    "post lockdown estimates": post_lockdown_estimates,                                            
    }
    
    return direct_estimation_results
    

def indirect_parameter_estimation(
                                    stocks: dict,
                                    direct_parameter_estimates: dict, 
                                    beta_CC: float,
                                    beta_S: float
    ) -> dict:
    """ Performs indirect parameter estimation.

    This function indirectly estimates some of the parameters from the processed data. 

    Parameters
    ----------
        stocks: dict
            Dictionary containing the initial conditions of the stocks in 
            the given epoch.
        direct_parameter_estimates:
            Dictionary of directly estimated parameters. 
        beta_CC:
            Proportoin of consultant appointments with referral to consultant. 
        beta_S:   
            Inflow of patients to the symptomatic population.
    Returns
    ----------
       parameters_values: dict
            Dictionary containing the value of each parameter.
    """

    # This is all in the paper.   
    beta_CT =  (1/direct_parameter_estimates["alpha_C"])*(direct_parameter_estimates["beta_M"]*stocks["P_T"] + direct_parameter_estimates["alpha_T"])

    beta_DC = (1/direct_parameter_estimates["alpha_D"])*(direct_parameter_estimates["beta_M"]*stocks["P_C"]\
                + direct_parameter_estimates["alpha_C"]*(1 - beta_CC) \
                - direct_parameter_estimates["alpha_G"]*direct_parameter_estimates["beta_GC"] - direct_parameter_estimates["alpha_T"])

    beta_DG = 1 - beta_DC

    beta_SG = (1/stocks["P_S"])*(direct_parameter_estimates["beta_M"]*stocks["P_G"]\
                + direct_parameter_estimates["alpha_G"] - direct_parameter_estimates["alpha_D"]*beta_DG\
                - direct_parameter_estimates["alpha_C"]*direct_parameter_estimates["beta_CG"])
            
    beta_GS = (1/direct_parameter_estimates["alpha_G"])*(stocks["P_S"]*(beta_SG + direct_parameter_estimates["beta_M"]) - beta_S)                
    
    beta_GD =  (1 - direct_parameter_estimates["beta_GC"] - beta_GS)
                                
    beta_CD = (1/direct_parameter_estimates["alpha_C"])*(direct_parameter_estimates["beta_M"]*stocks["P_D"] +\
                direct_parameter_estimates["alpha_D"] - direct_parameter_estimates["alpha_G"]*beta_GD)

    parameters_values = {
                    "beta_M": direct_parameter_estimates["beta_M"],
                    "beta_S": beta_S,
                    "alpha_G": direct_parameter_estimates["alpha_G"],
                    "alpha_D": direct_parameter_estimates["alpha_D"],
                    "alpha_C": direct_parameter_estimates["alpha_C"],
                    "alpha_T": direct_parameter_estimates["alpha_T"],
                    "beta_SG": beta_SG,
                    "beta_GC": direct_parameter_estimates["beta_GC"],
                    "beta_GD": beta_GD,
                    "beta_GS": beta_GS,
                    "beta_DC": beta_DC,
                    "beta_DG": beta_DG,
                    "beta_CT": beta_CT,
                    "beta_CD": beta_CD,
                    "beta_CC": beta_CC,
                    "beta_CG": direct_parameter_estimates["beta_CG"]
    }
    
    return parameters_values
    

def extract_stock_ground_truth(
                                data,
                                stock_name,
                                dates,
                                epoch_start_date,
                                epoch_end_date
    ):
    """ Extracts the ground truth for a given stock. 

    This function extracts the ground truth data for a given stock and 
    in a given epoch. We use this to optimise the model.

    Parameters
    ----------
        data: dict
            Dictionary containing the processed data. 
        stock_name: str
            Name of stock.
        dates: dict
            Dictionary of key dates. 
        epoch_start_date: str
            Start date of epoch.
        epoch_end_date
            End date of epoch.

    Returns
    ----------
        gt_data: array 
            Ground truth data for the stock.
   
    """
    # Compute the stock data using equations 29-33 ( page 71)
    stock_data = compute_stock_data(data, stock_name)
    
    # Determine start and end indices for the data. 
    non_zero_indices = np.nonzero(stock_data)
    data_start_index = np.min(non_zero_indices)
    data_end_index = np.max(non_zero_indices)

    # Determine the indices for the ground truth data. 
    gt_start_index = max(data_start_index,
                        (epoch_start_date - dates["start date"]).days)
                                 
    gt_end_index = min(data_end_index, 
                        (epoch_end_date - dates["start date"]).days)
    
    gt_data = stock_data[gt_start_index:gt_end_index]

    return gt_data

    
def extract_all_stocks_ground_truth(
                                data,
                                dates,
                                epoch_start_date,
                                epoch_end_date
    ):
    """ Extracts the ground truth for all the stocks. 

    This function extracts the ground truth data for all the stocks 
    in a given epoch. We use this to optimise the model.

    Parameters
    ----------
        data: dict
            Dictionary containing the processed data. 
        dates: dict
            Dictionary of key dates. 
        epoch_start_date: str
            Start date of epoch.
        epoch_end_date
            End date of epoch.

    Returns
    ----------
        gt_stocks: dict 
            Dictionary containing the ground truth data for each stock 
            in a given epoch. 
    """
    # Initialise dictionary. 
    gt_stocks = {
                    "P_G": None,
                    "P_D": None,
                    "P_C": None,
                    "P_T": None
    }
    
    # Compute ground truth stock data. 
    for stock_abbr in gt_stocks.keys():
        
        stock_name = stocks_plotting_names[stock_abbr]
    
        gt_stock = extract_stock_ground_truth(
                                    data,
                                    stock_name,
                                    dates,
                                    epoch_start_date,
                                    epoch_end_date
        )
        gt_stocks[stock_abbr] = gt_stock
        
    return gt_stocks


def pre_lockdown_function(
                            x,
                            gt_stocks,
                            stocks_initial_conditions,
                            direct_parameter_estimates, 
                            delta_1,
                            delta_2,
                            no_pre_lockdown_days,
                            time_step,
                            model, 
    ):
    """ Function to optimise in the pre-lockdown epoch.

    This function computes the loss between the model output and the 
    ground truth stocks for the pre-lockdown epoch. 

    Parameters
    ----------
        x: array
            Optimisation variables.
        gt_stocks: dict
            Dictionary containing the ground truth stocks.
        stocks_initial_conditions: dict
            Dictionary containing the initial conditions for the stocks.
        direct_parameter_estimates: dict
            Dictionary containing the direct parameter estimates. 
        delta_1: float
            Threshold for steady-state constraint.
        delta_2: float
            Threshold for proportions constraint.
        no_pre_lockdown_days: int
            Number of pre lockdown days.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.

    Returns
    ----------
        loss: float 
            RMSE in the pre lockdown epoch.
             
    """
    # Perform indirect parameter estimation.
    parameters_values = indirect_parameter_estimation(
                                    stocks_initial_conditions,
                                    direct_parameter_estimates, 
                                    x[0],
                                    x[1]
    )

    # Perform constraint checks.
    constraint_check = perform_constraint_checks(stocks_initial_conditions, parameters_values, delta_1, delta_2)
    
    # Set constraint penalty.
    constraint_penalty = 1e6*constraint_check

    # Set initial conditions.
    initial_conditions = set_initial_conditions(stocks_initial_conditions)
        
    # Set time step factor.
    time_step_factor = int(1/time_step)
    
    # Compute the total number of steps.
    no_time_steps = no_pre_lockdown_days*time_step_factor
    
    # Create parameter time series from parameter values.
    for key, value in parameters_values.items():
        parameters_values[key] = pd.Series(index=range(no_time_steps), data=value)
        
    parameters_time_series = set_parameter_time_series(parameters_values)

    # Run the ODE model.
    model_output = model.run( 
                                progress=False,
                                time_step=time_step,
                                final_time=no_pre_lockdown_days-1,
                                params=parameters_time_series,
                                initial_condition=(0, initial_conditions)
    )
    
    # Compute the RMSE between the model output for the stocks and the 
    # ground truth data for the stocks. 
    error = 0
    for stock_abbr in gt_stocks.keys():
        
        stock_name = stocks_plotting_names[stock_abbr]
        error += np.sqrt(mean_squared_error(gt_stocks[stock_abbr], model_output[stock_name]))
   
    loss = error + constraint_penalty
    
    return loss


def compute_final_conditions(
                                stocks_initial_conditions, 
                                direct_parameter_estimates,
                                epoch,
                                opt_results,
                                time_step, 
                                no_days,
                                model
    ):      
    """ Function to compute final conditions of a given epoch.

    This function computes the final conditions of a given epoch 
    using the parameter values obtained from the optimisation procedure
    for that epoch. 

    Parameters
    ----------
        stocks_initial_conditions: dict
            Dictionary containing the initial conditions for the stocks.
        direct_parameter_estimates: dict
            Dictionary containing the direct parameter estimates. 
        epoch: str
            String indicating the epoch.
        opt_results: dict
            Dictionary containing the results of the optimisation procedure.
        gt_stocks: dict
            Dictionary containing the ground truth stocks.
        time_step: int
            Time step for the ODE solver. 
        no_days: int
            Number of days in the epoch.
        model: 
            PYSD model.

    Returns
    ----------
        stocks_final_conditions: dict
            Dictionary containing the final condition values of the stocks.
        parameters_time_series: dict
            Dictionary containing the time series of each parameter.
        parameters_values: dict
            Dictionary containing the values of each parameter.
    """        
    if epoch=="pre lockdown":                
        # Perform indirect parameter estimation.
        parameters_values = indirect_parameter_estimation(
                                        stocks_initial_conditions,
                                        direct_parameter_estimates, 
                                        opt_results.x[0],
                                        opt_results.x[1]
        )
    else:
        parameters_values = {
                        "beta_M": direct_parameter_estimates["beta_M"],
                        "alpha_G": direct_parameter_estimates["alpha_G"],
                        "alpha_D": direct_parameter_estimates["alpha_D"],
                        "alpha_C": direct_parameter_estimates["alpha_C"],
                        "alpha_T": direct_parameter_estimates["alpha_T"],
                        "beta_GC": direct_parameter_estimates["beta_GC"],
                        "beta_CG": direct_parameter_estimates["beta_CG"],
                        "beta_S": opt_results.x[0],
                        "beta_SG": opt_results.x[1],
                        "beta_GS": opt_results.x[2],
                        "beta_GD": opt_results.x[3],
                        "beta_DG": opt_results.x[4],
                        "beta_DC": opt_results.x[5],
                        "beta_CD": opt_results.x[6],
                        "beta_CC": opt_results.x[7],
                        "beta_CT": opt_results.x[8],
        }
        
    # Set the initial conditions.
    initial_conditions = set_initial_conditions(stocks_initial_conditions)
        
    # Compute the time step factor.
    time_step_factor = int(1/time_step)
    
    # Compute the number of time steps.
    no_time_steps = no_days*time_step_factor
    
    # Create parameter time series from parameter values.
    parameters_time_series = {}
    
    for key, value in parameters_values.items():
        parameters_time_series[key] = pd.Series(index=range(no_time_steps), data=value)
        
    parameters_time_series = set_parameter_time_series(parameters_values)

    # Run model.
    model_output = model.run( 
                                progress=False,
                                time_step=time_step,
                                final_time=no_days-1,
                                params=parameters_time_series,
                                initial_condition=(0, initial_conditions)
    )
    
    # Extract final conditions. 
    stocks_final_conditions = {}
    
    for stock_abbr in stocks_initial_conditions.keys():

        stocks_final_conditions[stock_abbr] = model_output[stocks_plotting_names[stock_abbr]].iloc[-1]

    return stocks_final_conditions, parameters_time_series, parameters_values
    
  
def optimise_model_pre_lockdown(
                    data,
                    dates,
                    stocks_initial_conditions,
                    direct_parameter_estimates, 
                    delta_1, 
                    delta_2,
                    no_pre_lockdown_days,
                    time_step,
                    model,
                    max_iters
    ):
    """ Pre-lockdown epoch optimisation procedure.

    This function implements the pre-lockdown epoch optimisation procedure as detailed 
    in the paper. 

     
    Optimisation parameters
    ------------------------
    
    -- beta_CC
    -- beta_S

    Parameters
    ----------
        data: dict
            Dictionary containing the processed data. 
        dates:
            Dictionary containing the key dates. 
        stocks_initial_conditions: dict
            Dictionary containing the initial conditions for the stocks.
        direct_parameter_estimates: dict
            Dictionary containing the direct parameter estimates. 
        delta_1: float
            Threshold for steady-state constraint.
        delta_2: float
            Threshold for proportions constraint.
        no_pre_lockdown_days: int
            Number of pre lockdown days.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.
        max_iters: int 
            Integer indicating the maximum number of iterations for the optimisation 
            routine.

    Returns
    ----------
        pre_lockdown_results: dict 
            Dictionary of results.
             
    """
    # Define initial conditions for optimisation parameters.
    x_0 = [0, 100]
    
    # Define bounds for optimisation parameters.
    bounds = Bounds([0, 0], [1, np.power(10, 6)])
    
    # Define epoch start and end dates.
    epoch_start_date = dates["simulation start date"]
    epoch_end_date = dates["lockdown start date"]
    
    # Extract ground truth stock time-series for this epoch.
    gt_stocks = extract_all_stocks_ground_truth(
                                    data,
                                    dates,
                                    epoch_start_date,
                                    epoch_end_date
    )
    
    # Define optimisation arguments.
    arguments = (
                    gt_stocks,
                    stocks_initial_conditions,
                    direct_parameter_estimates, 
                    delta_1,
                    delta_2,
                    no_pre_lockdown_days,
                    time_step,
                    model
    )
    
    # Define optimisation options.
    options = {
                "maxiter": max_iters,
                "disp": True,
                "verbose": True,
    }
    
    
    # Run optimiser.
    opt_results = minimize(
                            pre_lockdown_function,
                            x_0,
                            args=arguments,
                            method='trust-constr',
                            options=options,
                            bounds=bounds
    )

    # Compute final conditions for the initial conditions of the next epoch.
    stocks_final_conditions, parameters_time_series, parameters_values = compute_final_conditions(
                                stocks_initial_conditions, 
                                direct_parameter_estimates,
                                "pre lockdown",
                                opt_results,
                                time_step, 
                                no_pre_lockdown_days,
                                model
    )     
    
    # Store and return results.
    pre_lockdown_results = {
                                "stocks initial conditions": stocks_initial_conditions,
                                "stocks final conditions": stocks_final_conditions,
                                "parameters values": parameters_values,
                                "parameters time series": parameters_time_series                        
    }
    
    
    return pre_lockdown_results


def compute_mean_squared_error(true: np.array, pred: np.array) -> float:
    """ Computes the mean squared error.
    
    Parameters
    ----------
        true: np.array
            Array containing the ground truth stock data.
        pred: np.array 
            Array containing the model output stock data.  
        
    Returns
    ----------
        rmse: float 
            Root mean squared error.
             
    """
    no_comparable_samples = min(len(true), len(pred))
    
    rmse = np.sqrt(mean_squared_error(true[0:no_comparable_samples], pred[0:no_comparable_samples]))

    return rmse
   
   
def post_lockdown_and_lockdown_function(
                        x,
                        gt_stocks,
                        stocks_initial_conditions,
                        direct_parameter_estimates, 
                        no_days,
                        time_step,
                        model, 
    ):
    """ Function to optimise in the lockdown and post-lockdown epoch.

    This function computes the loss between the model output and the 
    ground truth stocks for the lockdown and post-lockdown epoch. 

    Parameters
    ----------
        x: array
            Optimisation variables.
        gt_stocks: dict
            Dictionary containing the ground truth stocks.
        stocks_initial_conditions: dict
            Dictionary containing the initial conditions for the stocks.
        direct_parameter_estimates: dict
            Dictionary containing the direct parameter estimates. 
        no_days: int
            Number of days in the epoch.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.

    Returns
    ----------
        loss: float 
            RMSE in the epoch.
             
    """
    # Perform indirect parameter estimation.
    parameters_values = {
                    "beta_M": direct_parameter_estimates["beta_M"],
                    "alpha_G": direct_parameter_estimates["alpha_G"],
                    "alpha_D": direct_parameter_estimates["alpha_D"],
                    "alpha_C": direct_parameter_estimates["alpha_C"],
                    "alpha_T": direct_parameter_estimates["alpha_T"],
                    "beta_GC": direct_parameter_estimates["beta_GC"],
                    "beta_CG": direct_parameter_estimates["beta_CG"],
                    "beta_S": x[0],
                    "beta_SG": x[1],
                    "beta_GS": x[2],
                    "beta_GD": x[3],
                    "beta_DG": x[4],
                    "beta_DC": x[5],
                    "beta_CD": x[6],
                    "beta_CC": x[7],
                    "beta_CT": x[8],
    }

    # Set initial conditions.
    initial_conditions = set_initial_conditions(stocks_initial_conditions)

    # Set time step factor.    
    time_step_factor = int(1/time_step)
    
    # Compute the total number of steps.
    no_time_steps = no_days*time_step_factor
    
    # Create parameter time series from parameter values.
    for key, value in parameters_values.items():
        parameters_values[key] = pd.Series(index=range(no_time_steps), data=value)
        
    parameters_time_series = set_parameter_time_series(parameters_values)

    # Run the ODE model.
    model_output = model.run( 
                                progress=False,
                                time_step=time_step,
                                final_time=no_days-1,
                                params=parameters_time_series,
                                initial_condition=(0, initial_conditions)
    )
    
    # Compute the RMSE between the model output for the stocks and the 
    # ground truth data for the stocks. 
    error = 0
    for stock_abbr in gt_stocks.keys():
        
        stock_name = stocks_plotting_names[stock_abbr]
        error += compute_mean_squared_error(gt_stocks[stock_abbr], model_output[stock_name])

    loss = error 
    
    return loss


def optimise_model_lockdown(
                    data,
                    dates,
                    stocks_initial_conditions,
                    direct_parameter_estimates, 
                    delta_2, 
                    no_lockdown_days,
                    time_step,
                    model,
                    max_iters
    ):
    """ Lockdown epoch optimisation procedure.

    This function implements the lockdown epoch optimisation procedure as detailed 
    in the paper. 

    Optimisation parameters
    ------------------------
    
    -- beta_S
    -- beta_SG
    -- beta_GS
    -- beta_GD
    -- beta_DG
    -- beta_DC
    -- beta_CD
    -- beta_CC
    -- beta_CT

     
    Parameters
    ----------
        data: dict
            Dictionary containing the processed data. 
        dates:
            Dictionary containing the key dates. 
        stocks_initial_conditions: dict
            Dictionary containing the initial conditions for the stocks.
        direct_parameter_estimates: dict
            Dictionary containing the direct parameter estimates. 
        delta_2: float
            Threshold for proportions constraint.
        no_lockdown_days: int
            Number of lockdown days.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.
        max_iters: int 
            Integer indicating the maximum number of iterations for the optimisation 
            routine.

    Returns
    ----------
        lockdown_results: dict 
            Dictionary of results.
             
    """
    # Define initial conditions for optimisation parameters.
    x_0 = [0, 0, 0.3, 0.3, 0.5, 0.5, 0.25, 0.25, 0.25]
    
    # Define bounds for optimisation parameters.
    bounds = Bounds(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [np.power(10, 6), 1, 1, 1, 1, 1, 1, 1, 1, ]
    )
    
    # Compute direct parameter estimates.
    beta_GC = direct_parameter_estimates["beta_GC"]
    beta_CG = direct_parameter_estimates["beta_CG"]
  
    # Set proportoin constraints as linear program.
    lhs_constraint = [1-beta_GC-delta_2, 1-delta_2, 1-beta_CG-delta_2]
    rhs_constraint = [1-beta_GC+delta_2, 1+delta_2, 1-beta_CG+delta_2]
   
    matrix_constraint = [
                            [0, 0, 1, 1, 0, 0, 0, 0 ,0],
                            [0, 0, 0, 0, 1, 1, 0, 0 ,0],
                            [0, 0, 0, 0, 0, 0, 1, 1 ,1],
    ]
    

    linear_constraint = LinearConstraint(
                                            matrix_constraint,
                                            lhs_constraint,
                                            rhs_constraint
    )

    # Define epoch start and end dates.
    epoch_start_date = dates["lockdown start date"]
    epoch_end_date = dates["lockdown end date"]
    
    # Extract ground truth stock time-series for this epoch.
    gt_stocks = extract_all_stocks_ground_truth(
                                    data,
                                    dates,
                                    epoch_start_date,
                                    epoch_end_date
    )


    # Define optimisation arguments.
    arguments = (
                    gt_stocks,
                    stocks_initial_conditions,
                    direct_parameter_estimates, 
                    no_lockdown_days,
                    time_step,
                    model
    )
    
    # Define optimisation options.
    options = {
                "maxiter": max_iters,
                "disp": True,
                "verbose": True,
    }
    
    
    # Run optimiser.
    opt_results = minimize(
                            post_lockdown_and_lockdown_function,
                            x_0,
                            args=arguments,
                            method='trust-constr',
                            options=options,
                            bounds=bounds,
                            constraints=linear_constraint
    )

    # Compute final conditions for the initial conditions of the next epoch.
    stocks_final_conditions, parameters_time_series, parameters_values = compute_final_conditions(
                                stocks_initial_conditions, 
                                direct_parameter_estimates,
                                "lockdown",
                                opt_results,
                                time_step, 
                                no_lockdown_days,
                                model
    )     
    
    # Store and return results.
    lockdown_results = {
                            "stocks initial conditions": stocks_initial_conditions,
                            "stocks final conditions": stocks_final_conditions,
                            "parameters values": parameters_values,
                            "parameters time series": parameters_time_series                        
    }
    
    return lockdown_results


def optimise_model_post_lockdown(
                    data,
                    dates,
                    stocks_initial_conditions,
                    direct_parameter_estimates, 
                    delta_2, 
                    no_post_lockdown_days,
                    time_step,
                    model,
                    max_iters
    ):
    """ Post-lockdown epoch optimisation procedure.

    This function implements the post-lockdown epoch optimisation procedure as detailed 
    in the paper. 

    Optimisation parameters
    ------------------------
    
    -- beta_S
    -- beta_SG
    -- beta_GS
    -- beta_GD
    -- beta_DG
    -- beta_DC
    -- beta_CD
    -- beta_CC
    -- beta_CT

     
    Parameters
    ----------
        data: dict
            Dictionary containing the processed data. 
        dates:
            Dictionary containing the key dates. 
        stocks_initial_conditions: dict
            Dictionary containing the initial conditions for the stocks.
        direct_parameter_estimates: dict
            Dictionary containing the direct parameter estimates. 
        delta_2: float
            Threshold for proportions constraint.
        no_post_lockdown_days: int
            Number of post lockdown days.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.
        max_iters: int 
            Integer indicating the maximum number of iterations for the optimisation 
            routine.

    Returns
    ----------
        post_lockdown_results: dict 
            Dictionary of results.
             
    """
    # Define initial conditions for optimisation parameters.
    x_0 = [0, 0, 0.3, 0.3, 0.5, 0.5, 0.25, 0.25, 0.25]
    
    # Define bounds for optimisation parameters.
    bounds = Bounds(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [np.power(10, 6), 1, 1, 1, 1, 1, 1, 1, 1, ]
    )

    # Compute direct parameter estimates. 
    beta_GC = direct_parameter_estimates["beta_GC"]
    beta_CG = direct_parameter_estimates["beta_CG"]
  
    # Set proportion constraints. 
    lhs_constraint = [1-beta_GC-delta_2, 1-delta_2, 1-beta_CG-delta_2]
    rhs_constraint = [1-beta_GC+delta_2, 1+delta_2, 1-beta_CG+delta_2]
    matrix_constraint = [
                            [0, 0, 1, 1, 0, 0, 0, 0 ,0],
                            [0, 0, 0, 0, 1, 1, 0, 0 ,0],
                            [0, 0, 0, 0, 0, 0, 1, 1 ,1],
    ]
    

    linear_constraint = LinearConstraint(
                                            matrix_constraint,
                                            lhs_constraint,
                                            rhs_constraint
    )

    # Define epoch start and end dates.
    epoch_start_date = dates["lockdown end date"]
    epoch_end_date = dates["post lockdown end date"]
    
    # Extract ground truth stock time-series for this epoch.
    gt_stocks = extract_all_stocks_ground_truth(
                                    data,
                                    dates,
                                    epoch_start_date,
                                    epoch_end_date
    )    

    # Define optimisation arguments.
    arguments = (
                    gt_stocks,
                    stocks_initial_conditions,
                    direct_parameter_estimates, 
                    no_post_lockdown_days,
                    time_step,
                    model
    )
    
    # Define optimisation options.
    options = {
                "maxiter": max_iters,
                "disp": True,
                "verbose": True,
    }
    
    
    # Run optimiser.
    opt_results = minimize(
                            post_lockdown_and_lockdown_function,
                            x_0,
                            args=arguments,
                            method='trust-constr',
                            options=options,
                            bounds=bounds,
                            constraints=linear_constraint
    )

    # Compute final conditions for the initial conditions of the next epoch.
    stocks_final_conditions, parameters_time_series, parameters_values = compute_final_conditions(
                                stocks_initial_conditions, 
                                direct_parameter_estimates,
                                "post lockdown",
                                opt_results,
                                time_step, 
                                no_post_lockdown_days,
                                model
    )     
    
    # Store and return results.
    post_lockdown_results = {
                            "stocks initial conditions": stocks_initial_conditions,
                            "stocks final conditions": stocks_final_conditions,
                            "parameters values": parameters_values,
                            "parameters time series": parameters_time_series                        
    }
    
    return post_lockdown_results


def optimise_model_parameters(
                    dates,
                    data_dictionary,
                    time_periods,
                    gamma,
                    beta,
                    delta_1,
                    delta_2,
                    no_days_dictionary,
                    time_step,
                    model,
                    max_iters
    ):
    """ Optimise parameters. 

    This function implements all three optimisation procedures. 

    Parameters
    ----------
        data: dict
            Dictionary containing the processed data. 
        dates: dict
            Dictionary containing the key dates. 
        time_periods: dict
            Dictionary containing the indices of the time periods. 
        gamma: float
            Difference in mortality rates between Wales and England. 
        beta: float 
            Ratio of people registered at a GP practice in Wales v England.
        delta_1: float
            Threshold for steady-state constraint.
            delta_2: float
            Threshold for proportions constraint.
        no_days_dictionary: dict
            Dictionary containing the number of days in each epoch.
        time_step: int
            Time step for the ODE solver. 
        model: 
            PYSD model.
        max_iters: int 
            Integer indicating the maximum number of iterations for the optimisation 
            routine.

    Returns
    ----------
        post_lockdown_results: dict 
            Dictionary of results.
             
    """
    # Compute data statistics
    data_statistics = compute_all_data_statistics(data_dictionary, time_periods)

    # Perform direct parameter estimation from data. 
    direct_estimation_results = direct_parameter_estimation(
                                                        data_statistics,
                                                        gamma,
                                                        beta
    )
    
    # Estimate initial conditions for the stocks in the pre lockdown epoch..
    pre_lockdown_initial_conditions = compute_initial_conditions(data_statistics["pre lockdown"])
    
    # Extract processed data from total data dictionary.
    data = extract_processed_data(data_dictionary)

    # Perform pre-lockdown optimisation procedure.
    pre_lockdown_results = optimise_model_pre_lockdown(
                                    data,
                                    dates,
                                    pre_lockdown_initial_conditions,
                                    direct_estimation_results["pre lockdown estimates"],
                                    delta_1, 
                                    delta_2,
                                    no_days_dictionary["pre lockdown"],
                                    time_step,
                                    model,
                                    max_iters
    )
    
    print("Fitted pre lockdown")
    
    # Perform lockdown optimisation procedure.
    lockdown_results = optimise_model_lockdown(
                                    data,
                                    dates,
                                    pre_lockdown_results["stocks final conditions"],
                                    direct_estimation_results["lockdown estimates"],
                                    delta_1, 
                                    no_days_dictionary["lockdown"],
                                    time_step,
                                    model,
                                    max_iters
    )
    
    print("Fitted lockdown")

    # Perform post lockdown optimisation procedure.
    post_lockdown_results = optimise_model_post_lockdown(
                                    data,
                                    dates,
                                    lockdown_results["stocks final conditions"],
                                    direct_estimation_results["post lockdown estimates"],
                                    delta_1, 
                                    no_days_dictionary["post lockdown"],
                                    time_step,
                                    model,
                                    max_iters
    )
    print("Fitted postlockdown")
        
    # Store and save optimisation results.
    optimisation_results = {
                                "pre lockdown": pre_lockdown_results,
                                "lockdown": lockdown_results,
                                "post lockdown": post_lockdown_results,
    }
    
    np.save("optimisation_results.npy", optimisation_results, allow_pickle="True")
    

def extract_output_statistics(model_outputs: dict):
    """ Extract model output statistics.

    This function extracts the mean, lb and ub time series for each stock 
    from the model outputs.
    
    Parameters
    ----------
       model_outputs: dict
            Dictionary containing the model outputs for each stock.
       
    Returns
    ----------

        model_outputs: dict
            Dictionary containing the model output statistics for each stock.
       
    """

    for key, value in model_outputs.items():

    
        mean_value = np.mean(value, 0)
        std_value = np.std(value, 0)
        
        ub_value = np.max(value, 0)
        lb_value = np.min(value, 0)
  
        model_outputs[key] = {
                                "lower bound": mean_value - std_value, 
                                "upper bound": mean_value + std_value,
                                "mean": mean_value
        }

    return model_outputs


def compute_stock_data(data: dict, stock_name: str) -> np.array:
    """ Compute the stock data. 

    This function computes the stock data for the ground truth stock 
    values using equations 29-33 (page 71). 
    
    Parameters
    ----------
        data: dict 
            Dictionary containing the processed data. 
        stock_name: str
            String indicating which stock to compute. 

    Returns
    ----------
        stock_data: np.array
            Stock data. 
    
    """
    # Compute stock data. 
          
    if stock_name == "Symptomatic population":
    
        stock_data = data["p_conds"]*data["N_L_E"]
        
    elif stock_name == "GP waiting list population":
        
        stock_data = data["p_conds"]*data["B_GP"] 
    
    elif stock_name == "Diagnostic waiting list population":
        
        stock_data = data["W_echocardiography"] + data["W_electrophysiology"] \
                   + data["W_MRI_cardiology"] + data["W_CT_cardiology"]

    elif stock_name == "Consultant waiting list population":

        stock_data = data["W_incomplete_c"] + data["W_incomplete_cs"] 
        
    elif stock_name == "Treatment waiting list population":
        
        stock_data = data["W_incomplete_dta_c"] + data["W_incomplete_dta_cs"]
    
    return stock_data


def extract_ground_truth_data(stock_data: np.array, dates: dict):
    """ Extract the ground truth data.

    This function just extracts the ground truth data for the pre-lockdown, 
    lockdown and post-lockdown epochs and puts this all in one array for each 
    stock.

    Parameters
    ----------
        stock_data: np.array
            Array containing the stock data.  
        dates: dict
            Dictionary containing key dates.
    Returns
    ----------
        gt_data: np.array
            Ground truth data for plotting.
        gt_dates: list
            Ground truth data dates for plotting.
    
    """
    # Compute non-zero stock data indices.
    non_zero_indices = np.nonzero(stock_data)
    data_start_index = np.min(non_zero_indices)
    data_end_index = np.max(non_zero_indices)

    # Find the start and end indices for the simulatin period excluding the projection 
    # epoch.
    gt_start_index = max(data_start_index,
                        (dates["simulation start date"] - dates["start date"]).days)
                                 
    gt_end_index = min(data_end_index, 
                        (dates["post lockdown end date"] - dates["start date"]).days)
    

    no_gt_days = gt_end_index - gt_start_index

    gt_start_date = dates["start date"] + timedelta(days=int(gt_start_index))
    
    gt_data = stock_data[gt_start_index:gt_end_index]
    gt_dates = [gt_start_date + timedelta(days=x) for x in range(no_gt_days)]

    return gt_data, gt_dates


def plot_model_output(model_plotting_parameters: dict, stock_name: str):
    """ Plots model output for one stock.

    Parameters
    ----------
        model_plotting_parameters: dict
            Dictionary of plotting parameters..  
        stock_name: str
            Stock name string.
    """
    # Extract data from plotting parameter dictionary.
    data = model_plotting_parameters["data"]
    dates = model_plotting_parameters["dates"]
    model_outputs_dict = model_plotting_parameters["model outputs"]
    save_flag = model_plotting_parameters["save flag"]
    show_flag = model_plotting_parameters["show flag"]
    text_fontsize = model_plotting_parameters["text fontsize"]
    ticks_fontsize = model_plotting_parameters["ticks fontsize"]
    figure_label = model_plotting_parameters["figure label"]

    # Compute chosen stock data. 
    stock_data = compute_stock_data(data, stock_name)

    # Extract the stock data in between the simulation start date and the post-lockdown end date.
    gt_data, gt_dates = extract_ground_truth_data(stock_data, dates)
    
    # Initialise plots. 
    fig, axes = plt.subplots(1, 1)

    model_output_colours = {
                                "0": "royalblue",
                                "5": "forestgreen",
                                "10": "darkgoldenrod",
                                "25": "forestgreen",
                                "50": "darkgoldenrod",                                
    }

    data_labels = []
    data_lines = []
    
    # Determine the maximum value for the y-axis to compute order of magnitude 
    # for the ticks.
    max_y_value = np.max(gt_data)
    min_y_value = np.min(gt_data)
    
    for key, model_outputs in model_outputs_dict.items():
       
        model_mean = model_outputs[stock_name]["mean"]
        model_ub = model_outputs[stock_name]["upper bound"]
        model_lb = model_outputs[stock_name]["lower bound"]
        
        max_y_value = max(max_y_value, np.max(model_ub))
        min_y_value = min(min_y_value, np.min(model_lb))

    order_magnitude = np.floor(np.log10(np.max(max_y_value)))
 
    # Plot rescaled ground truth stock data. 
    data_line_1 = axes.plot(gt_dates, gt_data/np.power(10, order_magnitude), "red", lw=2)
       
    # Plot rescaled model output data. 
    for key, model_outputs in model_outputs_dict.items():
        
        model_mean = model_outputs[stock_name]["mean"]/np.power(10, order_magnitude)
        model_ub = model_outputs[stock_name]["upper bound"]/np.power(10, order_magnitude)
        model_lb = model_outputs[stock_name]["lower bound"]/np.power(10, order_magnitude)
        
        data_line_2 = axes.plot(dates["simulation dates"], model_mean, model_output_colours[key], lw=2)
        
        axes.plot(dates["simulation dates"], model_ub,  model_output_colours[key], alpha=1,lw=2)
        axes.plot(dates["simulation dates"], model_lb,  model_output_colours[key], alpha=1,lw=2)
        axes.fill_between(dates["simulation dates"], model_lb, model_ub, alpha=0.3, color= model_output_colours[key])
        
        data_lines.append(data_line_2[0])
        
        if key=="0":
            data_labels.append("Model: no intervention")
        elif key=="5":
            data_labels.append("Model: 5% increase")
        elif key=="10":
            data_labels.append("Model: 10% increase")
        elif key=="25":
            data_labels.append("Model: 25% increase")
        elif key=="50":
            data_labels.append("Model: 50% increase")

    data_labels.reverse()
    data_lines.reverse()
    
    data_lines.insert(0,data_line_1[0])
    data_labels.insert(0, "Ground truth")

    # Set title and axis labels. 
    epoch_colours = {
                                    "lockdown": "k",
                                    "projection period": "blueviolet"
    }

    # Generate the epoch rule sets which just signify which colour
    epoch_labels = generate_epoch_labels(dates)
    
    # Add intervention lines.
    axes, lines, labels = add_intervention_lines(
                                                    epoch_labels,
                                                    epoch_colours,
                                                    axes
    )
    # Plot the ticks.
    tick_date_formatting = "%m-%Y"
        
    # Plot ticks.
    no_days_available = len(dates["simulation dates"])
    
    plot_ticks(
                    ticks_fontsize,
                    tick_date_formatting,
                    dates["simulation dates"],
                    order_magnitude,                
                    no_days_available,
    )
   
    # Set axes labels.
    axes.set_xlabel("Date", fontsize=text_fontsize)
    axes.set_ylabel("People", fontsize=text_fontsize)

    # Set axes bounds.
    plt.gca().set_xbound(dates["simulation dates"][0], dates["simulation dates"][-1]+timedelta(7)) 
    
    # Plot legend.
    fig.legend(data_lines, data_labels,bbox_to_anchor = (0.9, 1),
              ncol=2, prop={"size":20})

    try:
        os.mkdir(os.getcwd() + "/figures/model_output/")
    except:
        pass

    try:
        os.mkdir(os.getcwd() + "/figures/model_output/" + figure_label + "/")
    except:
        pass
    
    # Compute figure name for saving purposes.
    figure_name = os.getcwd() + "/figures/model_output/" + figure_label + "/" + stock_name 
        
    # Format the figure and save or show as requested.
    format_figure(fig, lines, labels, figure_name, save_flag, show_flag)
    
    plt.close("all")


def plot_all_model_outputs(model_plotting_parameters: dict):
    """ Plot all model outputs.

    Parameters
    ----------
        model_plotting_parameters: dict
            Dictionary of plotting parameters..  

    """
    
    # Extract simulation data. 
    model_plotting_parameters["data"] = extract_processed_data(model_plotting_parameters["data"])

    for stock_name in model_plotting_parameters["stock names"]:

        plot_model_output(model_plotting_parameters, stock_name)


def generate_epoch_labels(dates: dict, lockdown_only=False):
    
    lockdown_rule_set =  {
                        "start_date": dates["lockdown start date"],
                        "end_date": dates["lockdown end date"],
                        "label": "lockdown"
    }
    
    projection_period_rule_set =  {
                        "start_date": dates["projection start date"] + timedelta(3),
                        "end_date": dates["projection end date"],
                        "label": "projection period"
    }
    
    if lockdown_only:
        
        return [lockdown_rule_set]
    
    else:
        
        return [lockdown_rule_set, projection_period_rule_set]


def perform_constraint_checks(
                                stocks: dict,
                                parameters: dict, 
                                ss_threshold: float,
                                proportions_threshold: float, 
    ):
    """ Performs constraint checks in the pre-lockdown epoch. 

    This function performs the constraint checks for the pre-lockdown
    optimisation. Details can be found in the paper. 
    
    Parameters
    ----------
        stocks: dict
            Dictionary containing the stock values.
        parameters: dict 
            Dictionary containing the parameter values.
        ss_threshold: float 
            Threshold value for the steady-state condition. 
        proportions_threshold: float 
            Threshold value for the proportions condition.
        
    
    Returns
    ----------
        constraint_check: float
            Binary value indicating whether all constraints are satisfied or not.
       
    """
    # Perform non-negativity check ons tocks and variables. 
    non_neg_check = perform_non_negativity_check(stocks, parameters)

    # Perform proportions checks.
    gp_prop_check = perform_proportions_check(
                                                [ 
                                                    parameters["beta_GS"],
                                                    parameters["beta_GC"],
                                                    parameters["beta_GD"]
                                                ],
                                                proportions_threshold
    )

    diagnostics_prop_check = perform_proportions_check(
                                                [ 
                                                    parameters["beta_DC"],
                                                    parameters["beta_DG"]
                                                ],
                                                proportions_threshold
    )

    consultant_prop_check = perform_proportions_check(
                                                [ 
                                                    parameters["beta_CT"],
                                                    parameters["beta_CD"],
                                                    parameters["beta_CG"],
                                                    parameters["beta_CC"],
                                                    
                                                ],
                                                proportions_threshold
    )   

    # Perform steady-state checks.
    ss_check = perform_steady_state_checks(stocks, parameters, ss_threshold)

    # Collect conditions in a list.
    conditions = [
                    non_neg_check,
                    gp_prop_check,
                    diagnostics_prop_check,
                    consultant_prop_check, 
                    ss_check
    ]

    # Compute valid model check.
    constraint_check = np.sum(conditions)!=5

    return constraint_check


def perform_steady_state_check(
                                stock_value: float,
                                ss_error: float,
                                ss_threshold: float
    ):
    """ Determines if steady-state constraint is satisfied. 

    This function determines if the steady-state constraint is satisfied 
    for a given stock and constraint threshold value.
    
    Parameters
    ----------
        stock_value: float 
            Value of stock.
        ss_error: float 
            Steady-state assumption error. 
        ss_threshold: float
            Steady-state constraint threshold.
    
    Returns
    ----------
        satisfied: bool
            Boolean indicating whether the steady-state constraint 
            is satisfied or not. 
    """
    satisfied = False

    if ss_error!=0.0:
        satisfied = ss_error/stock_value < ss_threshold
    else:
        satisfied = True

    return satisfied


def treatment_waiting_list_check(   
                                    stocks: dict,
                                    variables: dict,
                                    ss_threshold: float 
    ):
    """ Steady-state check for the treatment waiting list stock.
    
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        variables: dict
            Dictionary containing the variables.  
        ss_threshold: float
            Steady-state constraint threshold value.
    
    Returns
    ----------
        TL_ssc: bool 
            Boolean indicating whether the steady-state constraint is 
            satisfied or not.
       
    """  
    # Compute the steady-state error.
    dP_T =  abs(variables["alpha_C"]*variables["beta_CT"] \
           - variables["beta_M"]*stocks["P_T"] - variables["alpha_T"])
    
    # Check if the error is small enough to satisfy the steady-state assumption
    TL_ssc = perform_steady_state_check(stocks["P_T"], dP_T, ss_threshold)
    
    return TL_ssc


def consultant_waiting_list_check(   
                                    stocks: list,
                                    variables: list,
                                    ss_threshold: float
    ):
    """ Steady-state check for the consultant waiting list stock.
    
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        variables: dict
            Dictionary containing the variables.  
        ss_threshold: float
            Steady-state constraint threshold value.
    
    Returns
    ----------
        CL_ssc: bool 
            Boolean indicating whether the steady-state constraint is 
            satisfied or not.
    """
    # Compute the steady-state error.
    dP_C = abs(variables["alpha_G"]*variables["beta_GC"] \
          + variables["alpha_D"]*variables["beta_DC"] \
          + variables["alpha_T"] - variables["beta_M"]*stocks["P_C"]\
          - variables["alpha_C"]*(variables["beta_CD"] \
                             + variables["beta_CT"] \
                             + variables["beta_CG"]))

    # Check if the error is small enough to satisfy the steady-state assumption
    CL_ssc = perform_steady_state_check(stocks["P_C"], dP_C, ss_threshold)
    
    return CL_ssc


def gp_waiting_list_check(   
                            stocks: list,
                            variables: list,
                            ss_threshold: float
    ):
    """ Steady-state check for the GP waiting list stock.
    
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        variables: dict
            Dictionary containing the variables.  
        ss_threshold: float
            Steady-state constraint threshold value.
    
    Returns
    ----------
        GP_ssc: bool 
            Boolean indicating whether the steady-state constraint is 
            satisfied or not.
    """ 
    # Compute the steady-state error.
    dP_G = abs(stocks["P_S"]*variables["beta_SG"] \
           + variables["alpha_D"]*variables["beta_DG"]\
           + variables["alpha_C"]*variables["beta_CG"]\
           - variables["beta_M"]*stocks["P_G"] - variables["alpha_G"])

    # Check if the error is small enough to satisfy the steady-state assumption
    GPL_ssc = perform_steady_state_check(stocks["P_G"], dP_G, ss_threshold)
    
    return GPL_ssc


def symptomatic_population_check(   
                                    stocks: list,
                                    variables: list,
                                    ss_threshold: float
    ):
    """ Steady-state check for the sympomatic population stock.
    
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        variables: dict
            Dictionary containing the variables.  
        ss_threshold: float
            Steady-state constraint threshold value.
    
    Returns
    ----------
        S_ssc: bool 
            Boolean indicating whether the steady-state constraint is 
            satisfied or not.
    """ 
    # Compute the steady-state error.
    dP_S = abs(variables["beta_S"] + variables["alpha_G"]*variables["beta_GS"]\
         - stocks["P_S"]*variables["beta_SG"]\
         - variables["beta_M"]*stocks["P_S"])
  
    #stock_average/ss_error < ss_threshold)
    # Check if the error is small enough to satisfy the steady-state assumption
    S_ssc = perform_steady_state_check(stocks["P_S"], dP_S, ss_threshold)
    
    return S_ssc


def diagnostic_waiting_list_check(   
                                    stocks: list,
                                    variables: list,
                                    ss_threshold: float
    ):
    """ Steady-state check for the diagnostic waiting list stock.
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        variables: dict
            Dictionary containing the variables.  
        ss_threshold: float
            Steady-state constraint threshold value.
    
    Returns
    ----------
        DL_ssc: bool 
            Boolean indicating whether the steady-state constraint is 
            satisfied or not.
    """ 
    # Compute the steady-state error.
    dP_D = abs(variables["alpha_G"]*variables["beta_GD"]\
          + variables["alpha_C"]*variables["beta_CD"]\
          - variables["beta_M"]*stocks["P_D"] - variables["alpha_D"])

    # Check if the error is small enough to satisfy the steady-state assumption
    DL_ssc = perform_steady_state_check(stocks["P_D"], dP_D, ss_threshold)
    
    return DL_ssc


def perform_steady_state_checks(
                                    stocks: dict,
                                    variables: dict,
                                    ss_threshold: float
    ):
    """ Steady-state check for all relevant stocks.
    
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        variables: dict
            Dictionary containing the variables.  
        ss_threshold: float
            Steady-state constraint threshold value.
    
    Returns
    ----------
        steady_state_check: bool 
            Boolean indicating whether the steady-state constraint is 
            satisfied for all relevant stocks.
    """ 
    # Perform steady-state checks.
    S_ssc = symptomatic_population_check(stocks, variables, ss_threshold)
    GPL_ssc = gp_waiting_list_check(stocks, variables, ss_threshold)
    DL_ssc = diagnostic_waiting_list_check(stocks, variables, ss_threshold)
    CL_ssc = consultant_waiting_list_check(stocks, variables, ss_threshold)
    TL_ssc = treatment_waiting_list_check(stocks, variables, ss_threshold)

    # Compute the total steady-state check.
    steady_state_check = sum([S_ssc, GPL_ssc, DL_ssc, CL_ssc, TL_ssc]) == 5

    return steady_state_check
   

def perform_proportions_check(proportions_list: list, proportions_threshold: float):
    """ Computes the proportions check.

    This function determines if the proportion parameters sum to 1 for
    each stock.
    
    
    Parameters
    ----------
        proportions_list: list 
            List of proportions.
        proportions_threshold: float
            Threshold error.
    Returns
    ----------
        proportions_check: bool
            Boolean indicating if the proportions are within tolerance. 
    """
    # Compute the error in the sum of the proportions.
    proportions_error = abs(sum(proportions_list) - 1)

    # Check if the error is less than a specified threshold.
    proportions_check = proportions_error < proportions_threshold
    
    return proportions_check  


def perform_non_negativity_check(stocks: dict, parameters: dict):
    """ Checks if all stocks and parameters are non-negative.
    
    
    Parameters
    ----------
        stocks: dict 
            Dictionary containing the stocks.
        parameters: dict 
            Dictionary containing the paramaters.
        
    Returns
    ----------
        non_neg_check: bool 
            Boolean indicating whether all the stocks and parameters are non-negative 
            or not.
       
    """
    # Initialise check.
    non_neg_counter = 0
    
    # Check stocks.
    for stock in stocks.values():
        non_neg_counter += stock < 0
    
    # Check variables.
    for parameter in parameters.values():
        non_neg_counter += parameter < 0
        
    # Peform check
    non_neg_check = non_neg_counter == 0
    
    return non_neg_check



