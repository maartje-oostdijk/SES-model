# script written by Maartje Oostdijk, with help of Jan Kwakkel, exploration of single species-group mesopelagic SD model,
# EMA script from https://github.com/quaquel/epa1361_open/blob/master/Week%201-2%20-%20general%20intro%20to%20exploratory%20modelling/assignment%201%20-%20tutorial.ipynb
#some uncertainty analyses were taken from scripts written by Julie van Deelen: https://github.com/julievandeelen/Thesis-final
#modeling work done by MO, Laura Elsler, Willem Auping

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

# The mesopelagic fish model differential equations.

dSd_0 = None

def deriv(
    y,
    t,
    r,
    K,
    mm,
    mt,
    ft,
    rt,
    fr,
    sp,
    cd,
    td,
    tx,
    qc,
    uptake,
    q,
    scc,
    co2,
    pl,
    el,
    lobby,
    env,
):
    M, MM, FM, RM, S, E, C, SCC, TP = y  # all outcome variables
    global dSd_0

    # population dynamics of zooplankton & mesopelagic fish
    dMd_t = r * (1 - (M / K)) - mm * M - C  ##logistic growth function mesopelagic fish
    
    # carbon sequestration part mesopelagic fish
    dMMd_t = mm * M - M / mt  # deadfall MF
    dFMd_t = fr * r - M / ft  # fecal pellets MF
    dRMd_t = (1 - fr) * r - M / rt  # respiration MF
    
    # total seq
    dSd_t = dMMd_t + dFMd_t + dRMd_t
    # social cost of carbon
    dSCCd_t = S * scc * co2
    # profitability & effort
    prof = (sp - (cd / td)) * (1 - tx)# profitability is sale price -(cost per day devided by ton per day) times tax rate
    if (
        prof > 100
    ):  # fishery is only taking off if profits are beyond a certain level base this number on
        #e.g. blue whiting fishery profitability (Paoletti et al.2021)
        ur = uptake
    else:
        ur = 0
    dEd_t = prof * ur *0.125 #effort is profitability times uptake rate, 0.125 is conversion from carbon to live weight

    # governance section(protection for carbon seq. function comes into play after yearly carbon
    # seq becomes smaller than XX% of the initial value)
    
    if t==0:
        dSd_0 = dSd_t
    
    decision = 1
    if (
        prof > pl
    ):  # base this number (pl) on e.g. blue whiting fishery profitability (Paoletti et al.2021)
        decision = decision * lobby#industry lobby if profitability is high enough
    if dSd_t < dSd_0 * el:
        decision = decision * env#environmental protection once yearly rate of total sequestration by mesopelagic fish goes below a certain level
    q = q * decision

    dCd_t = min(E * qc * M, q * M)  # catch with maximum of catch being quota

    dTPd_t = prof*C*1000000000
    #dPd_t = prof*C
    
    return dMd_t, dMMd_t, dFMd_t, dRMd_t, dSd_t, dEd_t, dCd_t, dSCCd_t, dTPd_t

def M_model(
    K=0.3,
    r=4,
    M0=0.3,
    MM0=0,
    FM0=0,
    RM0=0,
    S0=0,
    E0=0,
    C0=0,
    TP0=0,
    SCC0=0,
    mm=0.67,
    mt=851,
    ft=599,
    rt=103,
    fr=0.25,
    sp=300,
    cd=17000,
    td=200,
    tx=0.2,
    qc=0.2,
    scc=100000000000,
    co2=3.67,
    pl=150000,#this parameter is not grounded in literature yet
    el=0.4,
    lobby=1.2,
    env=0.8,
    uptake=0.2,
    q=0.3,
    t=np.linspace(0, 50, 50),
):
    """

    Parameters
    ----------
    K : float
           Carrying capacity, based on level in Anderson et al., 2019
    r : float
            growth rate mesopelagic fish based on Tropfish R parameters for Maurolicus & Benthosema, & Fishbase
    mm : float
            mortality rate of mesopelagic fish based on Anderson et al., 2019
    rt : float
            sequestration time mesopelagic respiration, average time in Pinti et al., 2022
    ft : float
            sequestration time mesopelagic fecal pellets, average time in Pinti et al., 2022
    fr : float
            fraction to fecal of food conversion (Saba et al., 2021)
    mt : float
            sequestration time mesopelagic deadfall, average time in Pinti et al., 2022
    sp : float
            price for mesopelagic fish from Groeneveld et al. 2022
    cd : float
            cost per day Groeneveld et al., 2022; Kourantidou & Jin 2022
    td: float
        tonne per day, sumerised in Groeneveld et al. 2022
    tx: float
        tax rate, assumption, Groeneveld et al., 2022
    qc: float
    catchability, low in Kaartveld et al., 2012
    scc: float
    social cost of carbon, Interagency Working Group on Social Cost of Greenhouse Gases, 2021.
    co2: float
    conversion carbon to co2
    pl: float
        assumption/scenario
    el: float
        assumption/scenario
    M0 : int
         initial value mesopelagic fish
    MM0 : int
         carbon seq. mesopelagic fish
    t : ndarray
        points in time

    """

    # Initial conditions vector
    y0 = M0, MM0, FM0, RM0, S0, E0, C0, SCC0,TP0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(
        deriv,
        y0,
        t,
        args=(
            r,
            K,
            mm,
            mt,
            ft,
            rt,
            fr,
            sp,
            cd,
            td,
            tx,
            qc,
            uptake,
            q,
            scc,
            co2,
            pl,
            el,
            lobby,
            env,
        ),
    )
    M, MM, FM, RM, S, E, C, SCC,TP = ret.T

    return {
        "Mesofish": M,
        "MM": MM,
        "FM": FM,
        "RM": RM,
        "Total seq": S,
        "effort": E,
        "catch": C,
        "SCC": SCC,
        "total_profit":TP,
    }