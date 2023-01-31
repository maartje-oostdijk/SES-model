# script written by Maartje Oostdijk, with help of Jan Kwakkel, exploration of single species-group mesopelagic SD model,
# EMA script from https://github.com/quaquel/epa1361_open/blob/master/Week%201-2%20-%20general%20intro%20to%20exploratory%20modelling/assignment%201%20-%20tutorial.ipynb
#some uncertainty analyses were taken from scripts written by Julie van Deelen: https://github.com/julievandeelen/Thesis-final
#modeling work done by MO, Laura Elsler, Willem Auping

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

# The mesopelagic fish model differential equations.

dSd_0 = None #initial value of total sequestration rate

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
    #sp,
    #cd,
    #td,
    #tx,
    #pi,
    alpha,
    gamma,
    beta,
    cost,
    q,
    #uptake,
    q_a,
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
    dMd_t = r * M*(1 - (M / K)) - mm * M - C  ##logistic growth function mesopelagic fish #possibly change mm to m, and mesopelgic fish M to B?
    
    # have a metabolic rate for the entire population of mesopelagic fish, not base this on the growth rate!!!
    # carbon sequestration part mesopelagic fish
    dMMd_t = mm * M - MM / mt  # deadfall MF #ALL THE DECAY TERMS ARE WRONG!!!!!
    dFMd_t = fr * M * r- FM / ft  # fecal pellets  #DO THE DECAY TERMS NOW MAKE SENSE
    dRMd_t = (0.8 - fr) * M *r - RM / rt  # respiration MF #ALL THE DECAY TERMS ARE WRONG!!!! 0.9 is the assumed total carbon budget that does not go to growth but either to defecation or respiration, should be separate parameter.
    # using the growth rate here is tricky, as it also includes reproduction, will be good to ask someone more used to ecological modeling on how to do this
    #and fr should not just be multipled by M, but by some kind of intrinsic growth rate/metabolic rate
    
    # total seq
    dSd_t = dMMd_t + dFMd_t + dRMd_t
    # social cost of carbon
    dSCCd_t = S * scc * co2 #may be better change scc to something thats not the same as the outcome variable
    #dEd_t = E^(alpha*(gamma^(1/beta)*(q*E*M)^((beta-1)/beta)-cost*E)) #effort function from Fryxell
    dEd_t = E* np.exp(alpha*(pow(gamma,(1/beta))*pow((q*E*M),((beta-1)/beta))-cost*E)) #LAURA can hopefully help making sure I have the right notation in how to get the difference between the notation of dt+1 and the notation here that just needs to be dt, right? or no...
     #dEd_t = E^(a(y^(1/b)*(q*E*M)^((b-1)/b)-cost*E)) #effort function from Fryxell et al., 2017
 #et al., 2017
    # effort function
    #dEd_t = if ((sp*(E*q*M)-pi)/(cost*E)>0):
    #(sp*(E*q*M)-pi)/(cost*E)
    #else:
    #dEd_t = 0
     #cost is just continuous cost per unit effort, there is no set up cost included, pi according to OA theory is 0, but to make it more realistic we'll have a range, 
    # sp is sale price
    # profitability & effort
    #prof = (sp - (cd / td)) * (1 - tx)# profitability is sale price (euro/tonne) caught -(cost per day devided by ton per day) times tax rate #KICKOUTTAX!!!
    #if (
    #    prof > 100
   # ):  # fishery is only taking off if profits are beyond a certain level base this number on
        #e.g. blue whiting fishery profitability (Paoletti et al.2021)
    #    ur = uptake
    #else:
    #    ur = 0
    # dEd_t = prof * ur *0.125 #effor function please HJALP#effort is profitability times uptake rate, 0.125 is conversion from carbon to live weight

    # governance section(protection for carbon seq. function comes into play after yearly carbon
    # seq becomes smaller than XX% of the initial value)
    
    if t==0:
        dSd_0 = dSd_t#value of mesopelagic population when its not fished
    
    dTPd_t = 1*C-cost*dEd_t #is this the way to caluclate total profits of the fishery now? 1 is not correct as it should include price which should shift with profit
    
    decision_start = 1
    if (
        dTPd_t > pl
    ):  # base this number (pl) on e.g. blue whiting fishery profitability (Paoletti et al.2021)
        decision_l = decision_start * lobby#industry lobby if profitability is high enough
    else:
    	decision_l = decision_start
    if dSd_t < dSd_0 * el:
        decision_e = decision_l * env#environmental protection once yearly rate of total sequestration by mesopelagic fish goes below a certain level
    else:
    	decision_e = decision_l
   
    quota = q_a * decision_e #q_a is an adviced quota which is not yet defined decision is the multiplier because of lobby & environmental concern. 

    dCd_t = min(E * q * M, quota * M)  # catch with maximum of catch being quota
    

    #dTPd_t = prof*C*1000000000
    #dPd_t = prof*C
    
    return dMd_t, dMMd_t, dFMd_t, dRMd_t, dSd_t, dEd_t, dCd_t, dSCCd_t, dTPd_t

def M_model(
    M0=3,
    MM0=0,
    FM0=0,
    RM0=0,
    S0=0,
    E0=0.0001,
    C0=0,
    TP0=0,
    SCC0=0,
    K=3,
    r=4,
    mm=0.67,
    mt=851,
    ft=599,
    rt=103,
    fr=0.25,
    #sp=0.003,
    #pi = 0,
    cost= 0.4,
    alpha = 0.3,
    gamma = 0.01,
    beta = 1.5,
    #cd=17000,
    #td=200,
    #tx=0.2,
    q=0.002,
    scc=100000000000,
    co2=3.67,
    pl=150000,#this parameter is not grounded in literature yet
    el=0.4,
    lobby=1.2,
    env=0.8,
    #uptake=0.2,
    q_a = 0.3,
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
    
    q: float
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
    # Integrate the differential equations over the time grid, t. #what happens here
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
            #sp,
            #td,
            #cd
            #tx,
            q,#tonnes per day
            #pi,
            alpha,
            gamma,
            beta,
            cost,
            #uptake,
            q_a,
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
