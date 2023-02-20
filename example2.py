import numpy as np
from ema_workbench.connectors.pysd_connector import PysdModel

#from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS

from ema_workbench.analysis.plotting import lines, Density

def PredPrey(alpha=0.3, #effort scalar Fryxell et al., 2017
		gamma=200, #demand, as unit price when havest is 1, I multiplied this by 1,000,000,000 since our model is in giga ton but need to check with Laura
		beta=0.000014, #price sensitivity
             	q_e=200, #catchability 200 tones per unit effort (1 boat 1 day)
             	initial_meeso=2300000000,#*1000000000, #initial mesopelagic biomass, Anderson et al., 2019
             	initial_effort=0.010, #initial effort, mini amount of fishing in Norway, check if this times q times biomass would be reasonable level
             	initial_profit=0, #initial profit
             	initial_mort=0,
             	initial_harvest=1,
             	initial_fecal=0,
             	initial_resp=0,
             	initial_total=0,
             	initial_total_scc =0,
             	initial_mort_pr=0,
             	initial_fecal_pr=0,
             	initial_resp_pr=0,
             	initial_total_pr=0,
             	initial_total_scc_pr =0,
             	initial_cum_cost_harvest=0,
             	cost = 30000, #cost of unit effort 30,000$ for a day at sea
             	rmax = 1.2, #exponential growth rate mesopelagic fish, i.e. ln(normal growth rate)
             	m = 0.67, #mortality rate mesopelagic fish
             	K=3500000000,#*1000000000,#carrying capacity mesopelagic fish
             	#sp = 300*1000000000, #initial unit price mesopelagic harvest
             	pl = 10000, # profit level where mesopelagic fishing becomes interesting for lobby, placeholder(!)
             	lobby = 1.2, #lobby multiplier of quota
             	env = 0.8, #environmental protection multiplier of quota
             	el = 0.5,#loss of carbon sequestration ecosystem function at which decision maker becomes concerned
             	q_a=0.3, #adviced quota
             	mt=851,
             	ft=599,
             	rt=103,
             	fr = 0.2,
             	mb = 5,#metabolic rate, based on that a mesopelagic fish would eat its own bodyweight 5 times to replace its carbon in one year (Anderson et al., 2019)
             	cv = 7.7, #scalar from mesopelagic wet weigth to carbon Anderson et al., 2019
             	scc = 116, #000000000,#/1000000000,#check 116 *1000,000,000 ($ton * giga ton)
             	co2=3.67,#conversion from carbon to co2
             	dt=1, #timestep
             	final_time=50,
             	reps =1):

    #Initial values
    effort= np.zeros((reps, int(final_time/dt)+1))
    meeso = np.zeros((reps, int(final_time/dt)+1))
    prist_meeso_baseline = np.zeros((reps, int(final_time/dt)+1))
    profit = np.zeros((reps, int(final_time/dt)+1))
    harvest = np.zeros((reps, int(final_time/dt)+1))
    sim_time = np.zeros((reps, int(final_time/dt)+1))
    seq_mort = np.zeros((reps, int(final_time/dt)+1))
    seq_fecal = np.zeros((reps, int(final_time/dt)+1))
    seq_resp = np.zeros((reps, int(final_time/dt)+1))
    total_seq = np.zeros((reps, int(final_time/dt)+1))
    seq_mort_pr = np.zeros((reps, int(final_time/dt)+1))
    seq_fecal_pr = np.zeros((reps, int(final_time/dt)+1))
    seq_resp_pr = np.zeros((reps, int(final_time/dt)+1))
    total_seq_pr = np.zeros((reps, int(final_time/dt)+1))
    total_scc = np.zeros((reps, int(final_time/dt)+1))
    total_scc_pr = np.zeros((reps, int(final_time/dt)+1))
    cum_cost_harvest = np.zeros((reps, int(final_time/dt)+1))
    
    for r in range(reps):
        effort[r,0] = initial_effort
        meeso[r,0] = initial_meeso
        prist_meeso_baseline[r,0] = initial_meeso
        profit[r,0] = initial_profit
        harvest[r,0] = initial_harvest
        seq_mort[r,0] = initial_mort
        seq_fecal[r,0] = initial_fecal
        seq_resp[r,0] = initial_resp
        total_seq[r,0] = initial_total
        seq_mort_pr[r,0] = initial_mort_pr
        seq_fecal_pr[r,0] = initial_fecal_pr
        seq_resp_pr[r,0] = initial_resp_pr
        total_seq_pr[r,0] = initial_total_pr
        total_scc[r,0] = initial_total_scc
        total_scc_pr[r,0] = initial_total_scc_pr
        cum_cost_harvest[r,0] = initial_cum_cost_harvest

        #Calculate the time series
        for t in range(0, sim_time.shape[1]-1):

	#quota setting
            decision_start = 1
            if (profit[r,t] > pl and t >10):  # base this number (pl) on e.g. blue whiting fishery profitability (Paoletti et al.2021)
            	decision_l = decision_start * lobby#industry lobby if profitability is high enough
            else:
            	decision_l = decision_start
            if ((seq_resp[r,t]-seq_resp[r,t-1]) < seq_resp[r,10] * el and t>10):#laura please check if this makes sense
            	decision_e = decision_l * env
            else:#environmental protection once yearly rate of total sequestration by mesopelagic fish goes below a certain level
            	decision_e = decision_l #to implement decision_e check old model
   	    #q_a is an adviced quota which is not yet defined decision is the multiplier because of lobby & environmental concern. 
            quota = q_a * decision_e

            #if then else to make sure that they don't keep increasing effort when quota is reached
            p = gamma - gamma * beta * harvest[r, t]
            q = q_e/initial_meeso
            
            if ((effort[r,t] *  meeso[r,t] * q < quota * meeso[r,t]) and t >5): 
            	effort[r,t+1] = max(effort[r,t] + alpha*(p*q*effort[r,t]*meeso[r,t]-cost*effort[r,t]),0)#Fryxell et al 2017
            else: 
            	effort[r,t+1] = effort[r,t]
            	
            #calculate pristine baseline
            prist_meeso_baseline[r,t+1] = max(prist_meeso_baseline[r,t] * np.exp(rmax*(1-prist_meeso_baseline[r,t]/K))-m*prist_meeso_baseline[r,t],0)#no catches
            #rickert growth model:
            meeso[r,t+1] = max(meeso[r,t] * np.exp(rmax*(1-meeso[r,t]/K))-min(q*effort[r,t]*meeso[r,t],quota*meeso[r,t])-m*meeso[r,t],0)#harvest is capped by quota
            #harvest
            harvest[r, t+1] = harvest[r, t] + min(q*effort[r,t]*meeso[r,t],quota*meeso[r,t])

            #profit levels
            profit[r,t+1] = profit[r,t] + p*harvest[r, t+1]-effort[r,t]*cost
            
            #sequestration with harvesting
            seq_mort[r,t+1] = seq_mort[r,t]* (1-(1/mt)) + m*meeso[r,t]*cv
            #mortality * population level in previous time step minus total previous sequestration devided by time it takes to remineralise somehow this does not work well, the last term is resulting in a pattern in which there is basically no sequestration, which result in more sequestration in the fishing scenario versus the pristine without harvesting part of the model, and this is weird!
            seq_fecal[r,t+1] = seq_fecal[r,t]* (1-(1/ft)) +  fr * mb * meeso[r,t]*cv#same as above but then fraction of metabolic rate/losses going to fecal pellets
            seq_resp[r,t+1] = seq_resp[r,t]* (1-(1/rt)) +  (0.8-fr) * mb * meeso[r,t]*cv
            #same as above but then fraction of metabolic rate/losses going to respiration
            
            total_seq[r,t+1] = seq_mort[r,t+1]+seq_fecal[r,t+1]+seq_resp[r,t+1] 
            #+seq_resp[r,t+1] #add the different types of sequestration together
            
            #sequestration without harvesting/pristine population (ADD CONVERSION FROM WET WEIGHT TO CARBON)
            seq_mort_pr[r,t+1] = seq_mort_pr[r,t]* (1-(1/mt)) + m*prist_meeso_baseline[r,t]*cv 
            seq_fecal_pr[r,t+1] = seq_fecal_pr[r,t]* (1-(1/ft)) +  fr * mb * prist_meeso_baseline[r,t]*cv
            seq_resp_pr[r,t+1] = seq_resp_pr[r,t]* (1-(1/rt)) +  (0.8-fr) * mb * prist_meeso_baseline[r,t]*cv
            
            total_seq_pr[r,t+1] = seq_mort_pr[r,t+1]+seq_fecal_pr[r,t+1]+seq_resp_pr[r,t+1] 
            #add value in terms of social cost of carbon! and think of a way to calculate the cost fishing in social cost of carbon
            
            total_scc[r,t+1] = total_seq[r,t] * scc * co2 #total social cost of carbon, by multiplying total sequestration with cost of carbon and co2 conversion rate
            
            #baseline could be e.g. meeso + harvest versus meeso (which is meeso minus fishing) to calculate the scc cost
            #the cost of harvesting in social cost of carbon will only be calculated cumulatively
            total_scc_pr[r,t+1] = total_seq_pr[r,t] * scc * co2 #total sequestration in pristine state times social cost of carbon times co2 conversion
            
            cum_cost_harvest[r,t+1] =  total_scc_pr[r,t]-total_scc[r,t]#cumulative cost of harvesting measured in social cost of carbon
            
            sim_time[r,t+1] = (t+1)*dt
    
    #Return outcomes
    return {'TIME':sim_time,
            'effort':effort,
            'meeso':meeso,
            'catch':harvest,
            'profit':profit,
            'sequestration':total_seq,
            'sequestration_pristine':total_seq_pr,
            'total_social_cost_carbon':total_scc,
            'cumulative_social_cost':cum_cost_harvest}            
