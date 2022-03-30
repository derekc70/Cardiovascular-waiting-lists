"""
Python model 'model.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.functions import incomplete
from pysd.py_backend.statefuls import Integ

__pysd_version__ = "2.2.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent

_subscript_dict = {}

_namespace = {
    "TIME": "time",
    "Time": "time",
    "Symptomatic population": "symptomatic_population",
    "Symptom development rate": "symptom_development_rate",
    "Treatment rate": "treatment_rate",
    "Consultant discharge to primary care rate": "consultant_discharge_to_primary_care_rate",
    "Consultant referral to treatment rate": "consultant_referral_to_treatment_rate",
    "GP referral to consultant rate": "gp_referral_to_consultant_rate",
    "GP referral to diagnostics rate": "gp_referral_to_diagnostics_rate",
    "Consultant referral to diagnostic rate": "consultant_referral_to_diagnostic_rate",
    "Diagnostic discharge to GP rate": "diagnostic_discharge_to_gp_rate",
    "Diagnostic referral to consultant rate": "diagnostic_referral_to_consultant_rate",
    "GP discharge to symptomatic population rate": "gp_discharge_to_symptomatic_population_rate",
    "Consultant appointment supply": "consultant_appointment_supply",
    "Consultant waiting list death rate": "consultant_waiting_list_death_rate",
    "Diagnostic waiting list death rate": "diagnostic_waiting_list_death_rate",
    "Diagnostic appointment supply": "diagnostic_appointment_supply",
    "GP waiting list death rate": "gp_waiting_list_death_rate",
    "GP waiting list population": "gp_waiting_list_population",
    "GP appointment supply": "gp_appointment_supply",
    "GP appointment booking rate": "gp_appointment_booking_rate",
    "Treatment waiting list death rate": "treatment_waiting_list_death_rate",
    "Treatment appointment supply": "treatment_appointment_supply",
    "Proportion of GP appointments with discharge to symptomatic population": "proportion_of_gp_appointments_with_discharge_to_symptomatic_population",
    "Proportion of diagnostic appointments with discharge to GP": "proportion_of_diagnostic_appointments_with_discharge_to_gp",
    "Proportion of diagnostic appointments with consultant referral": "proportion_of_diagnostic_appointments_with_consultant_referral",
    "Proportion of consultant appointments with discharge to GP": "proportion_of_consultant_appointments_with_discharge_to_gp",
    "Proportion of consultant appointments with treatment referral": "proportion_of_consultant_appointments_with_treatment_referral",
    "Treatment waiting list population": "treatment_waiting_list_population",
    "Proportion of consultant appointments with diagnostic referral": "proportion_of_consultant_appointments_with_diagnostic_referral",
    "Proportion of GP appointments with consultant referral": "proportion_of_gp_appointments_with_consultant_referral",
    "Diagnostic waiting list population": "diagnostic_waiting_list_population",
    "Consultant waiting list population": "consultant_waiting_list_population",
    "Proportion of GP appointments with diagnostic referral": "proportion_of_gp_appointments_with_diagnostic_referral",
    "Died in symptomatic population": "died_in_symptomatic_population",
    "Died on GP waiting list population": "died_on_gp_waiting_list_population",
    "Died on consultant waiting list population": "died_on_consultant_waiting_list_population",
    "Died on diagnostic waiting list population": "died_on_diagnostic_waiting_list_population",
    "Died on treatment waiting list population": "died_on_treatment_waiting_list_population",
    "Probability of dying with symptoms": "probability_of_dying_with_symptoms",
    "Proportion of symptomatic patients booking GP appointments": "proportion_of_symptomatic_patients_booking_gp_appointments",
    "Symptomatic death rate": "symptomatic_death_rate",
    "FINAL TIME": "final_time",
    "INITIAL TIME": "initial_time",
    "SAVEPER": "saveper",
    "TIME STEP": "time_step",
}

_dependencies = {
    "symptomatic_population": {"_integ_symptomatic_population": 1},
    "symptom_development_rate": {
        "probability_of_dying_with_symptoms": 1,
        "consultant_waiting_list_population": 1,
        "diagnostic_waiting_list_population": 1,
        "gp_waiting_list_population": 1,
        "symptomatic_population": 1,
        "treatment_waiting_list_population": 1,
    },
    "treatment_rate": {
        "treatment_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
        "treatment_appointment_supply": 1,
    },
    "consultant_discharge_to_primary_care_rate": {
        "consultant_appointment_supply": 1,
        "proportion_of_consultant_appointments_with_discharge_to_gp": 2,
        "consultant_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "consultant_referral_to_treatment_rate": {
        "consultant_appointment_supply": 1,
        "proportion_of_consultant_appointments_with_treatment_referral": 2,
        "consultant_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "gp_referral_to_consultant_rate": {
        "gp_appointment_supply": 1,
        "proportion_of_gp_appointments_with_consultant_referral": 2,
        "gp_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "gp_referral_to_diagnostics_rate": {
        "gp_appointment_supply": 1,
        "proportion_of_gp_appointments_with_diagnostic_referral": 2,
        "gp_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "consultant_referral_to_diagnostic_rate": {
        "consultant_appointment_supply": 1,
        "proportion_of_consultant_appointments_with_diagnostic_referral": 2,
        "consultant_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "diagnostic_discharge_to_gp_rate": {
        "diagnostic_appointment_supply": 1,
        "proportion_of_diagnostic_appointments_with_discharge_to_gp": 2,
        "diagnostic_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "diagnostic_referral_to_consultant_rate": {
        "diagnostic_appointment_supply": 1,
        "proportion_of_diagnostic_appointments_with_consultant_referral": 2,
        "diagnostic_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "gp_discharge_to_symptomatic_population_rate": {
        "gp_appointment_supply": 1,
        "proportion_of_gp_appointments_with_discharge_to_symptomatic_population": 2,
        "gp_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "consultant_appointment_supply": {"consultant_appointment_supply": 1},
    "consultant_waiting_list_death_rate": {
        "consultant_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "diagnostic_waiting_list_death_rate": {
        "diagnostic_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "diagnostic_appointment_supply": {"diagnostic_appointment_supply": 1},
    "gp_waiting_list_death_rate": {
        "gp_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "gp_waiting_list_population": {"_integ_gp_waiting_list_population": 1},
    "gp_appointment_supply": {"gp_appointment_supply": 1},
    "gp_appointment_booking_rate": {
        "proportion_of_symptomatic_patients_booking_gp_appointments": 1,
        "symptomatic_population": 1,
    },
    "treatment_waiting_list_death_rate": {
        "treatment_waiting_list_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "treatment_appointment_supply": {"treatment_appointment_supply": 1},
    "proportion_of_gp_appointments_with_discharge_to_symptomatic_population": {},
    "proportion_of_diagnostic_appointments_with_discharge_to_gp": {},
    "proportion_of_diagnostic_appointments_with_consultant_referral": {},
    "proportion_of_consultant_appointments_with_discharge_to_gp": {},
    "proportion_of_consultant_appointments_with_treatment_referral": {},
    "treatment_waiting_list_population": {
        "_integ_treatment_waiting_list_population": 1
    },
    "proportion_of_consultant_appointments_with_diagnostic_referral": {},
    "proportion_of_gp_appointments_with_consultant_referral": {},
    "diagnostic_waiting_list_population": {
        "_integ_diagnostic_waiting_list_population": 1
    },
    "consultant_waiting_list_population": {
        "_integ_consultant_waiting_list_population": 1
    },
    "proportion_of_gp_appointments_with_diagnostic_referral": {},
    "died_in_symptomatic_population": {"_integ_died_in_symptomatic_population": 1},
    "died_on_gp_waiting_list_population": {
        "_integ_died_on_gp_waiting_list_population": 1
    },
    "died_on_consultant_waiting_list_population": {
        "_integ_died_on_consultant_waiting_list_population": 1
    },
    "died_on_diagnostic_waiting_list_population": {
        "_integ_died_on_diagnostic_waiting_list_population": 1
    },
    "died_on_treatment_waiting_list_population": {
        "_integ_died_on_treatment_waiting_list_population": 1
    },
    "probability_of_dying_with_symptoms": {},
    "proportion_of_symptomatic_patients_booking_gp_appointments": {},
    "symptomatic_death_rate": {
        "symptomatic_population": 1,
        "probability_of_dying_with_symptoms": 1,
    },
    "final_time": {},
    "initial_time": {},
    "saveper": {"time_step": 1},
    "time_step": {},
    "_integ_symptomatic_population": {
        "initial": {},
        "step": {
            "gp_discharge_to_symptomatic_population_rate": 1,
            "symptom_development_rate": 1,
            "gp_appointment_booking_rate": 1,
            "symptomatic_death_rate": 1,
        },
    },
    "_integ_gp_waiting_list_population": {
        "initial": {},
        "step": {
            "consultant_discharge_to_primary_care_rate": 1,
            "diagnostic_discharge_to_gp_rate": 1,
            "gp_appointment_booking_rate": 1,
            "gp_discharge_to_symptomatic_population_rate": 1,
            "gp_referral_to_consultant_rate": 1,
            "gp_referral_to_diagnostics_rate": 1,
            "gp_waiting_list_death_rate": 1,
        },
    },
    "_integ_treatment_waiting_list_population": {
        "initial": {},
        "step": {
            "consultant_referral_to_treatment_rate": 1,
            "treatment_rate": 1,
            "treatment_waiting_list_death_rate": 1,
        },
    },
    "_integ_diagnostic_waiting_list_population": {
        "initial": {},
        "step": {
            "consultant_referral_to_diagnostic_rate": 1,
            "gp_referral_to_diagnostics_rate": 1,
            "diagnostic_referral_to_consultant_rate": 1,
            "diagnostic_discharge_to_gp_rate": 1,
            "diagnostic_waiting_list_death_rate": 1,
        },
    },
    "_integ_consultant_waiting_list_population": {
        "initial": {},
        "step": {
            "diagnostic_referral_to_consultant_rate": 1,
            "gp_referral_to_consultant_rate": 1,
            "treatment_rate": 1,
            "consultant_discharge_to_primary_care_rate": 1,
            "consultant_referral_to_diagnostic_rate": 1,
            "consultant_referral_to_treatment_rate": 1,
            "consultant_waiting_list_death_rate": 1,
        },
    },
    "_integ_died_in_symptomatic_population": {
        "initial": {},
        "step": {"symptomatic_death_rate": 1},
    },
    "_integ_died_on_gp_waiting_list_population": {
        "initial": {},
        "step": {"gp_waiting_list_death_rate": 1},
    },
    "_integ_died_on_consultant_waiting_list_population": {
        "initial": {},
        "step": {"consultant_waiting_list_death_rate": 1},
    },
    "_integ_died_on_diagnostic_waiting_list_population": {
        "initial": {},
        "step": {"diagnostic_waiting_list_death_rate": 1},
    },
    "_integ_died_on_treatment_waiting_list_population": {
        "initial": {},
        "step": {"treatment_waiting_list_death_rate": 1},
    },
}

##########################################################################
#                            CONTROL VARIABLES                           #
##########################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 100,
    "time_step": lambda: 0.5,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


def final_time():
    """
    Real Name: FINAL TIME
    Original Eqn: 100
    Units: day
    Limits: (None, None)
    Type: constant
    Subs: None

    The final time for the simulation.
    """
    return __data["time"].final_time()


def initial_time():
    """
    Real Name: INITIAL TIME
    Original Eqn: 0
    Units: day
    Limits: (None, None)
    Type: constant
    Subs: None

    The initial time for the simulation.
    """
    return __data["time"].initial_time()


def saveper():
    """
    Real Name: SAVEPER
    Original Eqn: TIME STEP
    Units: day
    Limits: (0.0, None)
    Type: component
    Subs: None

    The frequency with which output is stored.
    """
    return __data["time"].saveper()


def time_step():
    """
    Real Name: TIME STEP
    Original Eqn: 0.5
    Units: day
    Limits: (0.0, None)
    Type: constant
    Subs: None

    The time step for the simulation.
    """
    return __data["time"].time_step()


##########################################################################
#                             MODEL VARIABLES                            #
##########################################################################


def symptomatic_population():
    """
    Real Name: Symptomatic population
    Original Eqn: INTEG ( GP discharge to symptomatic population rate+Symptom development rate-GP appointment booking rate-Symptomatic death rate, 0)
    Units: Patients
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_symptomatic_population()


def symptom_development_rate():
    """
    Real Name: Symptom development rate
    Original Eqn: Probability of dying with symptoms*(Consultant waiting list population+Diagnostic waiting list population+GP waiting list population+Symptomatic population+Treatment waiting list population)
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return probability_of_dying_with_symptoms() * (
        consultant_waiting_list_population()
        + diagnostic_waiting_list_population()
        + gp_waiting_list_population()
        + symptomatic_population()
        + treatment_waiting_list_population()
    )


def treatment_rate():
    """
    Real Name: Treatment rate
    Original Eqn: MAX(0,MIN(Treatment waiting list population*(1-Probability of dying with symptoms), Treatment appointment supply))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            treatment_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms()),
            treatment_appointment_supply(),
        ),
    )


def consultant_discharge_to_primary_care_rate():
    """
    Real Name: Consultant discharge to primary care rate
    Original Eqn: MAX(0, MIN( Consultant appointment supply * Proportion of consultant appointments with discharge to GP, Consultant waiting list population*(1-Probability of dying with symptoms)*Proportion of consultant appointments with discharge to GP ))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            consultant_appointment_supply()
            * proportion_of_consultant_appointments_with_discharge_to_gp(),
            consultant_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_consultant_appointments_with_discharge_to_gp(),
        ),
    )


def consultant_referral_to_treatment_rate():
    """
    Real Name: Consultant referral to treatment rate
    Original Eqn: MAX(0, MIN( Consultant appointment supply * Proportion of consultant appointments with treatment referral, Consultant waiting list population*(1-Probability of dying with symptoms)*Proportion of consultant appointments with treatment referral ))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            consultant_appointment_supply()
            * proportion_of_consultant_appointments_with_treatment_referral(),
            consultant_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_consultant_appointments_with_treatment_referral(),
        ),
    )


def gp_referral_to_consultant_rate():
    """
    Real Name: GP referral to consultant rate
    Original Eqn: MAX(0, MIN( GP appointment supply * Proportion of GP appointments with consultant referral, GP waiting list population*(1-Probability of dying with symptoms)*Proportion of GP appointments with consultant referral))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            gp_appointment_supply()
            * proportion_of_gp_appointments_with_consultant_referral(),
            gp_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_gp_appointments_with_consultant_referral(),
        ),
    )


def gp_referral_to_diagnostics_rate():
    """
    Real Name: GP referral to diagnostics rate
    Original Eqn: MAX(0, MIN( GP appointment supply * Proportion of GP appointments with diagnostic referral, GP waiting list population*(1-Probability of dying with symptoms)*Proportion of GP appointments with diagnostic referral))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            gp_appointment_supply()
            * proportion_of_gp_appointments_with_diagnostic_referral(),
            gp_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_gp_appointments_with_diagnostic_referral(),
        ),
    )


def consultant_referral_to_diagnostic_rate():
    """
    Real Name: Consultant referral to diagnostic rate
    Original Eqn: MAX(0, MIN( Consultant appointment supply * Proportion of consultant appointments with diagnostic referral, Consultant waiting list population*(1-Probability of dying with symptoms)*Proportion of consultant appointments with diagnostic referral ))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            consultant_appointment_supply()
            * proportion_of_consultant_appointments_with_diagnostic_referral(),
            consultant_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_consultant_appointments_with_diagnostic_referral(),
        ),
    )


def diagnostic_discharge_to_gp_rate():
    """
    Real Name: Diagnostic discharge to GP rate
    Original Eqn: MAX(0, MIN( Diagnostic appointment supply * Proportion of diagnostic appointments with discharge to GP, Diagnostic waiting list population*(1-Probability of dying with symptoms)*Proportion of diagnostic appointments with discharge to GP ))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            diagnostic_appointment_supply()
            * proportion_of_diagnostic_appointments_with_discharge_to_gp(),
            diagnostic_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_diagnostic_appointments_with_discharge_to_gp(),
        ),
    )


def diagnostic_referral_to_consultant_rate():
    """
    Real Name: Diagnostic referral to consultant rate
    Original Eqn: MAX(0, MIN( Diagnostic appointment supply * Proportion of diagnostic appointments with consultant referral, Diagnostic waiting list population*(1-Probability of dying with symptoms)*Proportion of diagnostic appointments with consultant referral ))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            diagnostic_appointment_supply()
            * proportion_of_diagnostic_appointments_with_consultant_referral(),
            diagnostic_waiting_list_population()
            * (1 - probability_of_dying_with_symptoms())
            * proportion_of_diagnostic_appointments_with_consultant_referral(),
        ),
    )


def gp_discharge_to_symptomatic_population_rate():
    """
    Real Name: GP discharge to symptomatic population rate
    Original Eqn: MAX(0, MIN( GP appointment supply *Proportion of GP appointments with discharge to symptomatic population, GP waiting list population(1-Probability of dying with symptoms)*Proportion of GP appointments with discharge to symptomatic population))
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        np.minimum(
            gp_appointment_supply()
            * proportion_of_gp_appointments_with_discharge_to_symptomatic_population(),
            gp_waiting_list_population(1 - probability_of_dying_with_symptoms())
            * proportion_of_gp_appointments_with_discharge_to_symptomatic_population(),
        ),
    )


def consultant_appointment_supply():
    """
    Real Name: Consultant appointment supply
    Original Eqn: Consultant appointment supply
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return consultant_appointment_supply()


def consultant_waiting_list_death_rate():
    """
    Real Name: Consultant waiting list death rate
    Original Eqn: MAX(0, Consultant waiting list population*Probability of dying with symptoms)
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0, consultant_waiting_list_population() * probability_of_dying_with_symptoms()
    )


def diagnostic_waiting_list_death_rate():
    """
    Real Name: Diagnostic waiting list death rate
    Original Eqn: MAX(0, Diagnostic waiting list population*Probability of dying with symptoms)
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0, diagnostic_waiting_list_population() * probability_of_dying_with_symptoms()
    )


def diagnostic_appointment_supply():
    """
    Real Name: Diagnostic appointment supply
    Original Eqn: Diagnostic appointment supply
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return diagnostic_appointment_supply()


def gp_waiting_list_death_rate():
    """
    Real Name: GP waiting list death rate
    Original Eqn: MAX(0, GP waiting list population*Probability of dying with symptoms)
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0, gp_waiting_list_population() * probability_of_dying_with_symptoms()
    )


def gp_waiting_list_population():
    """
    Real Name: GP waiting list population
    Original Eqn: INTEG ( Consultant discharge to primary care rate+Diagnostic discharge to GP rate+GP appointment booking rate-GP discharge to symptomatic population rate-GP referral to consultant rate-GP referral to diagnostics rate-GP waiting list death rate, 0)
    Units: Patients
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_gp_waiting_list_population()


def gp_appointment_supply():
    """
    Real Name: GP appointment supply
    Original Eqn: GP appointment supply
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return gp_appointment_supply()


def gp_appointment_booking_rate():
    """
    Real Name: GP appointment booking rate
    Original Eqn: MAX(0, Proportion of symptomatic patients booking GP appointments*Symptomatic population)
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0,
        proportion_of_symptomatic_patients_booking_gp_appointments()
        * symptomatic_population(),
    )


def treatment_waiting_list_death_rate():
    """
    Real Name: Treatment waiting list death rate
    Original Eqn: MAX(0, Treatment waiting list population*Probability of dying with symptoms)
    Units: Patients/day
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0, treatment_waiting_list_population() * probability_of_dying_with_symptoms()
    )


def treatment_appointment_supply():
    """
    Real Name: Treatment appointment supply
    Original Eqn: Treatment appointment supply
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return treatment_appointment_supply()


def proportion_of_gp_appointments_with_discharge_to_symptomatic_population():
    """
    Real Name: Proportion of GP appointments with discharge to symptomatic population
    Original Eqn: 0.01
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.01


def proportion_of_diagnostic_appointments_with_discharge_to_gp():
    """
    Real Name: Proportion of diagnostic appointments with discharge to GP
    Original Eqn: 0.1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.1


def proportion_of_diagnostic_appointments_with_consultant_referral():
    """
    Real Name: Proportion of diagnostic appointments with consultant referral
    Original Eqn: 0.1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.1


def proportion_of_consultant_appointments_with_discharge_to_gp():
    """
    Real Name: Proportion of consultant appointments with discharge to GP
    Original Eqn: 0.1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.1


def proportion_of_consultant_appointments_with_treatment_referral():
    """
    Real Name: Proportion of consultant appointments with treatment referral
    Original Eqn: 0.1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.1


def treatment_waiting_list_population():
    """
    Real Name: Treatment waiting list population
    Original Eqn: INTEG ( Consultant referral to treatment rate-Treatment rate-Treatment waiting list death rate, 0)
    Units: Patients
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_treatment_waiting_list_population()


def proportion_of_consultant_appointments_with_diagnostic_referral():
    """
    Real Name: Proportion of consultant appointments with diagnostic referral
    Original Eqn: 0.1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.1


def proportion_of_gp_appointments_with_consultant_referral():
    """
    Real Name: Proportion of GP appointments with consultant referral
    Original Eqn: 1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 1


def diagnostic_waiting_list_population():
    """
    Real Name: Diagnostic waiting list population
    Original Eqn: INTEG ( Consultant referral to diagnostic rate+GP referral to diagnostics rate-Diagnostic referral to consultant rate-Diagnostic discharge to GP rate-Diagnostic waiting list death rate, 0)
    Units: Patients
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_diagnostic_waiting_list_population()


def consultant_waiting_list_population():
    """
    Real Name: Consultant waiting list population
    Original Eqn: INTEG ( Diagnostic referral to consultant rate+GP referral to consultant rate+Treatment rate-Consultant discharge to primary care rate-Consultant referral to diagnostic rate-Consultant referral to treatment rate-Consultant waiting list death rate, 0)
    Units: Patients
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_consultant_waiting_list_population()


def proportion_of_gp_appointments_with_diagnostic_referral():
    """
    Real Name: Proportion of GP appointments with diagnostic referral
    Original Eqn: 1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 1


def died_in_symptomatic_population():
    """
    Real Name: Died in symptomatic population
    Original Eqn: INTEG ( Symptomatic death rate, 0)
    Units: Patients
    Limits: (0.0, None)
    Type: component
    Subs: None


    """
    return _integ_died_in_symptomatic_population()


def died_on_gp_waiting_list_population():
    """
    Real Name: Died on GP waiting list population
    Original Eqn: INTEG ( GP waiting list death rate, 0)
    Units: Patients
    Limits: (0.0, None)
    Type: component
    Subs: None


    """
    return _integ_died_on_gp_waiting_list_population()


def died_on_consultant_waiting_list_population():
    """
    Real Name: Died on consultant waiting list population
    Original Eqn: INTEG ( Consultant waiting list death rate, 0)
    Units: Patients
    Limits: (0.0, None)
    Type: component
    Subs: None


    """
    return _integ_died_on_consultant_waiting_list_population()


def died_on_diagnostic_waiting_list_population():
    """
    Real Name: Died on diagnostic waiting list population
    Original Eqn: INTEG ( Diagnostic waiting list death rate, 0)
    Units: Patients
    Limits: (0.0, None)
    Type: component
    Subs: None


    """
    return _integ_died_on_diagnostic_waiting_list_population()


def died_on_treatment_waiting_list_population():
    """
    Real Name: Died on treatment waiting list population
    Original Eqn: INTEG ( Treatment waiting list death rate, 0)
    Units: Patients
    Limits: (0.0, None)
    Type: component
    Subs: None


    """
    return _integ_died_on_treatment_waiting_list_population()


def probability_of_dying_with_symptoms():
    """
    Real Name: Probability of dying with symptoms
    Original Eqn: 0.01
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.01


def proportion_of_symptomatic_patients_booking_gp_appointments():
    """
    Real Name: Proportion of symptomatic patients booking GP appointments
    Original Eqn: 1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 1


def symptomatic_death_rate():
    """
    Real Name: Symptomatic death rate
    Original Eqn: MAX(0, Symptomatic population*Probability of dying with symptoms)
    Units: Patients/day
    Limits: (0.0, None)
    Type: component
    Subs: None


    """
    return np.maximum(
        0, symptomatic_population() * probability_of_dying_with_symptoms()
    )


_integ_symptomatic_population = Integ(
    lambda: gp_discharge_to_symptomatic_population_rate()
    + symptom_development_rate()
    - gp_appointment_booking_rate()
    - symptomatic_death_rate(),
    lambda: 0,
    "_integ_symptomatic_population",
)


_integ_gp_waiting_list_population = Integ(
    lambda: consultant_discharge_to_primary_care_rate()
    + diagnostic_discharge_to_gp_rate()
    + gp_appointment_booking_rate()
    - gp_discharge_to_symptomatic_population_rate()
    - gp_referral_to_consultant_rate()
    - gp_referral_to_diagnostics_rate()
    - gp_waiting_list_death_rate(),
    lambda: 0,
    "_integ_gp_waiting_list_population",
)


_integ_treatment_waiting_list_population = Integ(
    lambda: consultant_referral_to_treatment_rate()
    - treatment_rate()
    - treatment_waiting_list_death_rate(),
    lambda: 0,
    "_integ_treatment_waiting_list_population",
)


_integ_diagnostic_waiting_list_population = Integ(
    lambda: consultant_referral_to_diagnostic_rate()
    + gp_referral_to_diagnostics_rate()
    - diagnostic_referral_to_consultant_rate()
    - diagnostic_discharge_to_gp_rate()
    - diagnostic_waiting_list_death_rate(),
    lambda: 0,
    "_integ_diagnostic_waiting_list_population",
)


_integ_consultant_waiting_list_population = Integ(
    lambda: diagnostic_referral_to_consultant_rate()
    + gp_referral_to_consultant_rate()
    + treatment_rate()
    - consultant_discharge_to_primary_care_rate()
    - consultant_referral_to_diagnostic_rate()
    - consultant_referral_to_treatment_rate()
    - consultant_waiting_list_death_rate(),
    lambda: 0,
    "_integ_consultant_waiting_list_population",
)


_integ_died_in_symptomatic_population = Integ(
    lambda: symptomatic_death_rate(), lambda: 0, "_integ_died_in_symptomatic_population"
)


_integ_died_on_gp_waiting_list_population = Integ(
    lambda: gp_waiting_list_death_rate(),
    lambda: 0,
    "_integ_died_on_gp_waiting_list_population",
)


_integ_died_on_consultant_waiting_list_population = Integ(
    lambda: consultant_waiting_list_death_rate(),
    lambda: 0,
    "_integ_died_on_consultant_waiting_list_population",
)


_integ_died_on_diagnostic_waiting_list_population = Integ(
    lambda: diagnostic_waiting_list_death_rate(),
    lambda: 0,
    "_integ_died_on_diagnostic_waiting_list_population",
)


_integ_died_on_treatment_waiting_list_population = Integ(
    lambda: treatment_waiting_list_death_rate(),
    lambda: 0,
    "_integ_died_on_treatment_waiting_list_population",
)
