import utilities
import numpy as np
from matplotlib import pyplot as plt

# Set the dates
# Note: I've omitted the 20 here so 1 = 2001, 21 = 2021
collective_start_year = 1
collective_end_year = 21
collective_end_month = 12

# This is L from the paper (relevant for the primary care booking data)
max_delay = 60

# Initialise selenium browser.
browser = utilities.initialise_selenium_browser()

# Extract all data. 
utilities.extract_all_data(
                        browser,
                        collective_start_year,
                        collective_end_year,
                        collective_end_month, 
                        max_delay
)




