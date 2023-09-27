import numpy as np
from ema_workbench.connectors.pysd_connector import PysdModel

#from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS

from ema_workbench.analysis.plotting import lines, Density

def MEESO(alpha=0.3, #effort scalar Fryxell et al., 2017
		initial_gamma=350, #demand, as unit price when havest is 1,
		demand_mult = 1.004,
		beta=0.0005345, #price sensitivity
             	q_e=200, #catchability 200 tones per unit effort (1 boat 1 day)
             	initial_meeso=3000000000,#*1000000000, #initial mesopelagic biomass, Anderson et al., 2019
             	initial_effort=0.00010, #initial effort, mini amount of fishing in Norway, check if this times q times biomass would be reasonable level
             	initial_profit=0, #initial profit
             	initial_perc=0, #initial profit
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
             	rmax = 1.8, #exponential growth rate mesopelagic fish, i.e. ln(normal growth rate)
             	#m = 0.2, #mortality rate mesopelagic fish
             	K=3000000000,#*1000000000,#carrying capacity mesopelagic fish
             	#sp = 300*1000000000, #initial unit price mesopelagic harvest
             	pl = 10, # profit level where mesopelagic fishing becomes interesting for lobby, placeholder(!)
             	lobby = 1.2, #lobby multiplier of quota
             	env = 0.8, #environmental protection multiplier of quota
             	el = 0.5,#loss of carbon sequestration ecosystem function at which decision maker becomes concerned
             	q_a=0.3, #adviced quota
             	mt=851,
             	m=0,
             	ft=599,
             	rt=103,
             	#fr = 0.2,
             	m_f = 0.33,
             	f_f = 0.35,
             	r_f = 0.32,
             	#mb = 5,#metabolic rate, based on that a mesopelagic fish would eat its own bodyweight 5 times to replace its carbon in one year (Anderson et al., 2019)
             	cv = 0.77, #scalar from mesopelagic wet weigth to carbon injected Davison et al 2013
             	scc = 116, #000000000,#/1000000000,#check 116 *1000,000,000 ($ton * giga ton)
             	co2=3.67,#conversion from carbon to co2
             	dt=0.25, #timestep
             	final_time=12.5,
             	reps =1):

    #Initial values
    effort= np.zeros((reps, int(final_time/dt)+1))
    meeso = np.zeros((reps, int(final_time/dt)+1))
    prist_meeso_baseline = np.zeros((reps, int(final_time/dt)+1))
    profit = np.zeros((reps, int(final_time/dt)+1))
    gamma = np.zeros((reps, int(final_time/dt)+1))
    perc = np.zeros((reps, int(final_time/dt)+1))
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
        meeso[r,0] = K
        prist_meeso_baseline[r,0] = K
        profit[r,0] = initial_profit
        gamma[r,0] = initial_gamma
        perc[r,0] = initial_perc
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
            if (perc[r,t] > pl and t >10):  # base this number (pl) on e.g. blue whiting fishery profitability (Paoletti et al.2021)
            	decision_l = decision_start * lobby#industry lobby if profitability is high enough
            else:
            	decision_l = decision_start
            if ((seq_resp[r,t]-seq_resp[r,t-1]) < seq_resp[r,10] * el and t>10):#laura please check if this makes sense
            	decision_e = decision_l * env
            else:#environmental protection once yearly rate of total sequestration by mesopelagic fish goes below a certain level
            	decision_e = decision_l #to implement decision_e check old model
   	    #q_a is an adviced quota which is not yet defined decision is the multiplier because of lobby & environmental concern. 
            quota = q_a * decision_e
            #scenario with 20 million per year limit
            #IMPLEMENT SOME FLAG HERE TO SWITCH TO 'REALISTIC' SCENARIO
            #quota = 20000000/meeso[r,t]
            
            gamma[r, t+1]= gamma[r, t] * demand_mult #add a demand multiplier to the model of 1.2 and 1.5 based on increased aquaculture consumption and make Gamma into a variable not a float
            #if then else to make sure that they don't keep increasing effort when quota is reached
            if t<2 or q*effort[r,t]*meeso[r,t]<100:
            	p = gamma[r, t]
            else:
            	p = max(gamma[r, t] * (harvest[r, t]**(-1*beta)),0)
            
            #gamma*I(quantity^(-1*b)), data=trade, start = list(gamma=100,b=-0.14), trace = T) 
            
            q = q_e/K
            
            
            #profit levels
            profit[r,t+1] = p*q*effort[r,t]*meeso[r,t]-cost*effort[r,t]
            perc[r,t+1]= (p*harvest[r, t]-effort[r,t]*cost)/((effort[r,t]*cost)+0.0001)           
            #if ((effort[r,t] *  meeso[r,t] * q < quota * meeso[r,t]) and t >5): 
            #effort
            effort[r,t+1] = max(min(effort[r,t] + alpha*profit[r,t], (meeso[r,t]*quota)/q/meeso[r,t]),0.1)#Fryxell et al 2017

            #else: 
            #effort[r,t+1] = effort[r,t]	
            #if ((p*q*effort[r,t]*meeso[r,t]-cost*effort[r,t])<0): 
            	#effort[r,t+1] = 0#Fryxell et al 2017
           
            #calculate pristine baseline
            #prist_meeso_baseline[r,t+1] = max(prist_meeso_baseline[r,t] * np.exp(rmax*(1-prist_meeso_baseline[r,t]/K)),0)#no catches
            #rickert growth model:
            #meeso[r,t+1] = max(meeso[r,t] * np.exp(rmax*(1-meeso[r,t]/K))-min(q*effort[r,t]*meeso[r,t],quota*meeso[r,t]),0)#harvest is capped by quota
            
            #prist_meeso_baseline[r,t+1] = max(prist_meeso_baseline[r,t] * np.exp(rmax*(1-prist_meeso_baseline[r,t]/K))-m*prist_meeso_baseline[r,t],0)#no catches
            prist_meeso_baseline[r,t+1] = max(prist_meeso_baseline[r,t] + rmax *prist_meeso_baseline[r,t]*(1-(prist_meeso_baseline[r,t]/K)),0)
            #gordon schaefer surplus production model:
            #meeso[r,t+1] = max(meeso[r,t] * np.exp(rmax*(1-meeso[r,t]/K))-min(q*effort[r,t]*meeso[r,t],quota*meeso[r,t])-m*meeso[r,t],0)#harvest is capped by quota
            meeso[r,t+1] = max(meeso[r,t] + rmax * meeso[r,t]* (1-(meeso[r,t]/K))-harvest[r, t],0)#biomass + increase/decrease minus harvest 
            
            #harvest
            #harvest[r, t+1] = harvest[r, t] + q*effort[r,t]*meeso[r,t]
            harvest[r, t+1] = q*effort[r,t]*meeso[r,t]
            

            
            #sequestration with harvesting
            seq_mort[r,t+1] = seq_mort[r,t]* (1-(1/mt)) + meeso[r,t]*cv*m_f #sequestration by mortality, as a fraction of the population based on rates published in Davison et al. 2013
            seq_fecal[r,t+1] = seq_fecal[r,t]* (1-(1/ft)) +  meeso[r,t]*cv * f_f#same as above but then fraction of metabolic rate/losses going to fecal pellets
            seq_resp[r,t+1] = seq_resp[r,t]* (1-(1/rt)) +  meeso[r,t]*cv*r_f
            #same as above but then fraction of metabolic rate/losses going to respiration
           
            #add the different types of sequestration together:
            total_seq[r,t+1] = seq_mort[r,t+1]+seq_fecal[r,t+1]+seq_resp[r,t+1] 
            
            
            #sequestration without harvesting/pristine population (don't need this anymore given that initial mesopelagic is the same as carrying capacity)
            seq_mort_pr[r,t+1] = seq_mort_pr[r,t]* (1-(1/mt)) + prist_meeso_baseline[r,t]*cv*m_f
            seq_fecal_pr[r,t+1] = seq_fecal_pr[r,t]* (1-(1/ft)) +  prist_meeso_baseline[r,t]*cv * f_f
            seq_resp_pr[r,t+1] = seq_resp_pr[r,t]* (1-(1/rt)) +  prist_meeso_baseline[r,t]*cv*r_f
            
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
            'prist_meeso':prist_meeso_baseline,
            'catch':harvest,
            'sequestration':total_seq,
            'sequestration_pristine':total_seq_pr,
            'total_social_cost_carbon':total_scc,
            'cumulative_social_cost':cum_cost_harvest,
            'profit': profit,
            'gamma': gamma}            
