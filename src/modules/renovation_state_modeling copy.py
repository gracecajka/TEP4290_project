# %% [markdown]
# # Renovation state modelling

# %% [markdown]
# ### Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import decimal

from decimal import *
from scipy.stats import  norm, lognorm
from typing import List
import warnings

from dynamic_stock_model import DynamicStockModel

# %%

def get_renovation_cycles(time_interval:list, no_cycles:int, ren_mean_cycle:float, 
                          ren_rel_deviation:float,loc:float = 0, distribution: str = 'norm', show: bool = False):
    '''
    Arguments:
    - time_interval: list with first element as the starting time, 
      second element as the final time
    - no_cycles: how many renovation cycles (high numbers will be cut)
    - ren_mean_cycle: mean time between renovation cycles - basically 
      equivalent of the lifetimes of the componenets that get renovated.
      Note that for a lognormal distribution this is not really the mean
      - please check what it becomes!
    - ren_re_deviation: the spread of the renovation activity as the 
      relative(!) deviation from the mean in a normal distribution
    - loc: some extra offset if you want to
    - distribution: lognormal or normal. The meaning of the other 
      parameters changes when changing this.
    - show: whether or not to plot the renovation curve.
    '''
    
    times_t = np.linspace(time_interval[0], time_interval[1], time_interval[1]- time_interval[0]+1)
    ren_cycles = np.zeros(np.shape(times_t))
    ren_std_cycle = ren_mean_cycle*ren_rel_deviation
    cycle_index = 1
    while (cycle_index-1) < no_cycles and not (cycle_index * ren_mean_cycle > 1.5 * time_interval[1]): 
      if distribution == 'norm':
        single_renovation = norm.pdf(times_t,cycle_index*ren_mean_cycle, ren_std_cycle)
      elif distribution == 'lognorm':
        single_renovation =lognorm.pdf(times_t,s = 2*ren_rel_deviation, loc = cycle_index*ren_mean_cycle-2*ren_std_cycle+ times_t[0] + loc, scale = ren_mean_cycle-ren_std_cycle)
        #lognorm(x, s, loc, scale)
      ren_cycles += single_renovation
      cycle_index +=1

    if show:
      plt.figure(figsize=(12,8))
      if no_cycles >1:
        if distribution == 'norm':
          first_cycle = norm.pdf(times_t,ren_mean_cycle, ren_std_cycle)
        elif distribution == 'lognorm':
          print(f'we set s to {2*ren_rel_deviation}, loc to {ren_mean_cycle-2*ren_std_cycle}, scale to {ren_mean_cycle - ren_std_cycle}.')
          first_cycle =lognorm.pdf(times_t,s = 2*ren_rel_deviation, loc = ren_mean_cycle-2*ren_std_cycle + times_t[0] + loc, scale = ren_mean_cycle-ren_std_cycle)
        plt.plot(times_t,first_cycle,'-*', c = 'crimson', label = 'first single renovation cycle')
      plt.plot(times_t,ren_cycles, label = 'Renovation probability')
      plt.legend(loc = 'best')
      plt.title('Renovation profile for a single cohort')
      plt.xlabel('Time after construction') 
      plt.ylabel('Fraction renovated yearly')
      plt.show()

      if no_cycles>1:
        mean_time_first_cycle = np.einsum('i,i->', first_cycle, times_t)
      else:
        mean_time_first_cycle = np.einsum('i,i->', ren_cycles, times_t)
      print(f'the actual mean of the outflow time in the first renovation cycles is at {np.round(mean_time_first_cycle,1)} years.')
    
    return ren_cycles
    
    

# %% [markdown]
# ## Setting our renovation cycle to be only one lognormal one


# %%
class RenovationStock:
    '''
    Your new stock model! Expands your stock by one dimension for 
    renovation and assigns the correct amount (integers) of your stock
    to each renovation state
    '''

    def __init__(self, stock_original_tc_plus: np.ndarray, outflows_original_tc_plus:np.ndarray, 
                 ren_p_curves_t:List[np.ndarray], time_t: np.ndarray)->None:
        '''
        Initializes and already computes the stock.T

        Arguments:
        - stock_orginial_tc_plus: a stock matrix which should be 
          tc or higher - but you need to implement one for loop extra
          in the renovate function for everything beyond tc.
          Should be an np.ndarray filled with integer values
        - 
        '''
        self.s_o_tc_p = stock_original_tc_plus
        self.o_o_tc_p = outflows_original_tc_plus
        self.time_t = time_t
        self.hcs_t = [self.hazard_curve(ren_p_curve, show = True) for ren_p_curve in ren_p_curves_t]
        self.last_time_index = self.time_t[-1] - self.time_t[0]
        self.no_renovations = len(self.hcs_t)
        self.no_ren_states = self.no_renovations+1
    
    def renovate(self) -> None:
        '''
        adds the renovated stock to the attributes of the instance
        '''
        self.s_tc_p_r = self._extend_stock_like(self.s_o_tc_p, self.no_ren_states)
        #you could also store the outflows by renovation type.
        
        for time, time_slice in enumerate(self.s_o_tc_p):
            #add your additional dimensions (e.g. type) as ,: after time (but before the renovation state)
            self.s_tc_p_r[time, time, 0] = self.s_o_tc_p[time, time]
            for cohort, cohort_slice in enumerate(time_slice):
                #cohort_age = 
                #cohort_age = time - cohort
                cohort_age = time - cohort

                #print(f'the cohort age is {cohort_age}.')
                #or other cohort age definition if you want to change start date of renovation

                # we only check if there is a stock for this cohort at this time (cuts runtime in half!)
                if np.sum(cohort_slice) > 0: 
                    
                    #if cohort_age > len(self.hcs_t[0]):
                    #    raise Exception(f'your hazard function doesnt cover age {cohort_age}.')
                    #add another for loop per dimension beyond tc here
                    
                    for r in range(1, self.no_ren_states):
                        #split according to renovation
                        self.s_tc_p_r[time, cohort, :] = self._add_to_r(self.s_tc_p_r[time, cohort, :], cohort_age, r)

                    #dynamic modelling update (outflows across renovation states)
                    if time < self.last_time_index -1:
                        self.s_tc_p_r[time:time+2, cohort, :] = \
                            self._update_future_stock(self.s_tc_p_r[time:time+2, cohort, :], 
                                                      self.o_o_tc_p[time, cohort], time, cohort)
        self._check_results()
        return
    


    def hazard_curve(self, ren_prob_t:np.ndarray, dogmatic_stop:float = 0, 
                     show:bool = False)->np.ndarray:
        '''
        translates a renovation share curve to a hazard curve.
        
        Arguments:
        - ren_prob_t: the renovation share (probability) over time. 
          the sum of all entries must be 1!
        - dogmatic_stop: at which point should the survival function 
          be set to zero instead of a small fraction?
        - show: plot the curves if True
        '''
        #make the survival function
        survival_curve = np.zeros(np.shape(ren_prob_t))
        survival_curve[0] = 1
        for time, (sc_value, prob_value) in enumerate(zip(survival_curve[:-1], ren_prob_t)):
            survival_curve[time] = sc_value - prob_value
            survival_curve[time+1] = survival_curve[time]
        survival_curve = [0 if sc < dogmatic_stop else sc for sc in survival_curve]

        #derive the hazard curve from here
        hazard_curve = np.zeros(np.shape(ren_prob_t))
        hazard_curve[0] = 1 - survival_curve[0]
        for t_index, sc_value in enumerate(survival_curve[0:-1]):
            if sc_value != 0:
                hazard_curve[t_index+1] = (survival_curve[t_index]- survival_curve[t_index+1])/survival_curve[t_index]
                #if hazard_curve[t_index+1] > 0.3:
                #s    hazard_curve[t_index+1] = 0
            else:
                hazard_curve[t_index+1] = 0
        if show:
            fig = plt.figure(figsize=(12,8))
            ax1 = fig.subplots()
            ax1.plot(survival_curve,'--', c = 'crimson', label = 'survival function')
            ax1.plot(hazard_curve, '-.', c = 'forestgreen', label = 'hazard rate')
            ax1.set_xlabel('Time after manufacturing') 
            ax1.set_ylabel('Share still in use (survival)/probability of exting (hazard)')
            ax2 = ax1.twinx()
            ax2.plot(ren_prob_t, c = 'blue', lw = 2,  label = 'Share updated - PDF')
            plt.ylabel('Share updated of inflow')

            #make the legend for all
            lines = []
            labels = []
            for ax in fig.axes:
                axLine, axLabel = ax.get_legend_handles_labels()
                lines.extend(axLine)
                labels.extend(axLabel)
            fig.legend(lines, labels, loc = 'upper right')
            plt.title('Renovation/Updating - probabilities, survival, hazard')
            plt.show()
        return hazard_curve    
    
    def _extend_stock_like(self, stock_like:np.ndarray, no_ren_states:int)->np.ndarray:
        '''returns extended zero-array with one extra dimension no_renovations entries'''
        #extend original matrix dimensions
        old_shape = np.shape(stock_like)
        new_shape = list(old_shape)
        new_shape.append(no_ren_states)
        new_shape = tuple(new_shape)
        
        #make new container
        return np.zeros(new_shape)
    
    def _add_to_r(self, renovation_patch_r:np.ndarray, age_of_patch:int, r_index:int)->np.ndarray:
        '''
        Only updates the one patch for only the one type of renovation 
        (and only from renovation levels below).

        Arguments:
        - renovation_patch_r: slice of the total stock that needs an 
          update (one dimensional array: no_renovations)
        - age_of_patch: age (time-cohort) of that patch
        - r_index: which level of renovation
        '''
        renovated_amount = [int(self.hcs_t[r_index-1][age_of_patch] * stock) for stock in renovation_patch_r[:r_index]]
        for ren_level in range(r_index):
            renovation_patch_r[ ren_level] -=renovated_amount[ren_level]
        renovation_patch_r[r_index] += sum(renovated_amount)
        return renovation_patch_r
    
    def _update_future_stock(self, t1t2_stock_tr:np.ndarray, current_outflows:int, 
                             time:int, cohort:int) -> np.ndarray:
        '''
        Brings the current stock into the next year and takes of outflows.
        Arguments:
        - t1t2_stock_tr: slice of the stock with all renovation
          levels both at the current and the next instant.
        - current_outflows: total outflows from stock this instant
        - time: for correction of the outflows
        - cohort: for correction of the ouflows
        - type : you will have to put type in here as well.
        '''
        # if we neglected some outflows the year before, we just add the difference to the ouflows
        current_outflows +=np.sum(t1t2_stock_tr[0,:]) - self.s_o_tc_p[time, cohort]

        total_stock = np.sum(t1t2_stock_tr[0,:])
        if int(current_outflows) == 0 or total_stock < 1:
            t1t2_stock_tr[1,:] = t1t2_stock_tr[0,:]
        else:
            #print(f'The total stock is {total_stock}., the current stock_r is \
            #       {[t1t2_stock_tr[0,r] for r in range(self.no_ren_states)]}, \
            #       the current outflows are{current_outflows}.' )
            outflows_r = [int(t1t2_stock_tr[0,r]/total_stock * current_outflows) for r in range(self.no_ren_states)]
            trys = 0
            while round(sum(outflows_r),2)!=round(current_outflows, 2):
                if trys < 4:
                    #print(f'in the {trys} iteration the outflows are {outflows_r}.')
                    if sum(outflows_r) != 0:
                        outflows_r = list(np.rint(np.einsum('i,->i',outflows_r,current_outflows/sum(outflows_r))))
                    else: #if currently all elements of outflows are 0
                        outflows_r[0] = current_outflows
                elif trys < 9: #we just give the the difference to the biggest outflows
                    #print(f'the current outflows are {current_outflows}, the sum of \
                    # the ourflows_r is {sum(outflows_r)}, coming from the outflows {outflows_r}.')
                    difference = current_outflows - sum(outflows_r)

                    index_max = outflows_r.index(max(outflows_r))
                    if isinstance(index_max, int):
                        index_max = [index_max]
                    #print(f'the difference is {difference}, len(index_max) is {len(index_max)}.')
                    min_change = int(difference/len(index_max))
                    extra_change = difference%len(index_max) #might not split perfectly -> to first state
                    for counter, max_index in enumerate(index_max):
                        outflows_r[max_index] += min_change
                        if counter == 0:
                            outflows_r[max_index] += extra_change
                else:
                    break 
                    #raise Exception(f'ehm...the while loop for adjusting outflows failed.')
                trys +=1
            t1t2_stock_tr[1,:] = t1t2_stock_tr[0,:]        
            for state in range(self.no_ren_states):
                t1t2_stock_tr[1,state] = t1t2_stock_tr[1,state] - outflows_r[state]
                if t1t2_stock_tr[1,state] <0:
                    #if the new stock is negative we adjust the current stock such that the 
                    # outflows are redistributed but the stock of that state is not negative
                    eligible_indices = list(np.linspace(state, self.no_ren_states, self.no_ren_states - state, endpoint = False))
                    for index, stock in enumerate(t1t2_stock_tr[1,:state]):
                        if stock>0:
                            eligible_indices.append(index)
                    eligible_indices.sort()
                    
                    t1t2_stock_tr[1,:] = self.resolve_negative(t1t2_stock_tr[1,:], state, state, eligible_indices)

        return t1t2_stock_tr
                
    def resolve_negative(self, lo_values:np.ndarray, negative_index:int, 
                              current_state:int , eligible_indices:list)->np.ndarray:
        '''
        Deals with negative values for integer stocks - 
        no need to change this or to understand it even.
        '''
        no_eligible = len(eligible_indices)    
        split_negative = int(lo_values[negative_index]/no_eligible)
        extra = lo_values[negative_index]%no_eligible
        for counter, index in enumerate(eligible_indices):
            index = int(index)
            lo_values[index] += split_negative
            if counter == 0:
                lo_values[index] += extra
        lo_values[negative_index] = 0
        for index, element in enumerate(lo_values[:current_state]):
            if element < 0:
                lo_values = self.resolve_negative(lo_values, index, current_state, eligible_indices)

        return lo_values
    
    def plot_renovation_states_cohort(self, cohorts:list)->None:
        data_c_tr = np.zeros((len(cohorts), np.shape(self.s_tc_p_r)[0],np.shape(self.s_tc_p_r)[-1] ))
        
        for index, c in enumerate(cohorts):
            data_c_tr[index, :, :] = self.s_tc_p_r[:,c,:]

        self._plot_renovation_states(data_c_tr)
        

    def plot_renovation_total_stock(self)->None:
        self._plot_renovation_states(data_p_tr = np.array( [np.sum(self.s_tc_p_r, axis = 1)]))
    
    def _plot_renovation_states(self, data_p_tr:np.ndarray)->None:
        for data_tr in data_p_tr:
            #print(data_tr)
            plt.figure(figsize=(12,8))
            for ren in range(self.no_ren_states):
                plt.plot(self.time_t, data_tr[:,ren], 's-', label = f'renovation state {ren}')
            plt.plot(self.time_t, np.sum(data_tr[:,:], axis = 1), label = 'total stock from sum of renovation states')
            if np.shape(data_p_tr)[0] == 1: #if it is the total stock (hopefully)
                #if you have types, add a slice here and write axis = (1,2)
                plt.plot(self.time_t, np.sum(self.s_o_tc_p[:,:], axis = 1), ls = 'dotted', label = 'total stock from original stock')
            plt.margins(0,0)
            plt.title('Renovation split')
            plt.xlabel('Year') 
            plt.ylabel('number (not) updated')
            plt.legend(loc = "best")
            plt.margins(0.01, 0.01)
            plt.show()
        return
    
    def _check_results(self)->None:
        '''Does some checking with hints.'''
        #check that nd.arrays contain ints:
        if not np.sum(self.s_o_tc_p)%1 == 0:
            warnings.warn('Seems like your input stock contains non integers')
        
        times_stock_wrong = 0
        for time, time_slice in enumerate(self.s_tc_p_r):
            difference_ren_orig = np.sum(time_slice) - np.sum(self.s_o_tc_p[time])
            if difference_ren_orig/np.sum(time_slice) > 0.05:
                warnings.warn(f'Your renovation stock in year {time} is more than 5% different to the original.')
                times_stock_wrong +=1
            if times_stock_wrong > 10:
                warnings.warn(f'Your total stock was off for more than 10 years - we disabled warnings')
                break
        
        return
    
# %% [markdown]
# ## Example on the vehicle fleet stock

# %% [markdown]
# ## adjust EI_cjr to EI_tcjr

# %%
class EnergyIntensity:
    '''extends EI to a time dimension'''

    def __init__(self, EI_c_p_r:np.ndarray, time_of_state:int,  time_t:np.ndarray, adjustment_parameters:List[list], ) -> None:
        '''
        Does everything for you by default.

        Arguments:
        - EI_c_p_r: for example EI_cjr
        - time_of_state: time at which you know the EI_cjr
        - time_t: the full time you want to consider, so we can define
          EI_tcjr for the entire time_range
        - adjustment_parameters: list (len = r) of lists of start and 
        '''
        self.time_t = time_t
        self.EI_c_p_r = EI_c_p_r
        self.EI_tc_p_r = self._extend_EI_dims(self.EI_c_p_r, self.time_t)
        self.t_o_state = time_of_state
        
        self.adjustment_parameters = adjustment_parameters
        self.no_renovations = len(self.adjustment_parameters)
        self.adjustment_curves = self.get_adjustment_curves()
        self.adjust_EI()
        return

    
    def adjust_EI(self) -> None:
        '''populates self.EI_tc_p_r.'''
        for c_index, cohort_EI in enumerate(self.EI_c_p_r):
            cohort_age_2020 =  self.t_o_state - self.time_t[0] - c_index
            if cohort_age_2020 < 0:
                #for future cohorts, we assume we know the efficiences at age 0
                cohort_age_2020 = 0
            #add for loop here if needed for other dimensions
            for ren_index, ren_state_EI in enumerate(cohort_EI):
                #rescale adjustment curve:
                adjustment_curve = self.adjustment_curves[ren_index]
                #print(f'the adjustment curve is of type{type(adjustment_curve)}, and has vlaues {adjustment_curve}.')
                #rescale such that EI is rescaled with 1 at the age it has in self.t_o_state
                adjustment_curve = np.einsum('i, ->i', adjustment_curve, 1/adjustment_curve[cohort_age_2020])
                #print(f'the rensate EI is of type {type(ren_state_EI)} and has value {ren_state_EI}.')
                self.EI_tc_p_r[c_index:,c_index, ren_index] = np.einsum('i,->i' ,adjustment_curve, ren_state_EI)[0:len(self.time_t)-c_index]

        return
    
    def _extend_EI_dims(self, old_EI:np.ndarray, time_t:np.ndarray)->np.ndarray:
        '''returns extended zero-array with time dimension at first'''
        #extend original matrix dimensions
        old_shape = np.shape(old_EI)
        new_shape = [np.shape(time_t)[0]] +list(old_shape)
        new_shape = tuple(new_shape)
        
        #make new container
        return np.zeros(new_shape)


    def get_adjustment_curves(self, show = False) -> List[np.ndarray]:
        '''
        Makes a curve that will be used for adjusting EI in time.
        You probably want to change this mehtod and make it more
        advanced to reflect some of your choices or intuitions.
        '''
        adjustment_curves = [np.linspace(p[0], p[1],np.shape(self.time_t)[0]) for p in self.adjustment_parameters]

        if show:
            for ren in range(self.no_renovations):
                plt.figure(figsize=(12,8))
                plt.plot(adjustment_curves[ren])
                plt.xlabel('age of the cohort')
                plt.ylabel('adjustment (%) for the EI')
                plt.title('first adjustment curve')
                plt.show()

        return adjustment_curves
    
    def plot_some_EIs(self, cohorts:list)->None:
        plt.figure(figsize=(12,8))
        for c  in cohorts:
            for ei in [0,1,2]:
                plt.plot(self.time_t, self.EI_tc_p_r[:,c,ei], '-', label = f'EI ren state {ei} of cohort {c+self.time_t[0]}')
                plt.plot(max(self.t_o_state, c + self.time_t[0]), self.EI_c_p_r[c,ei], '*' , label = f'original data ren state {ei} of cohort {c+self.time_t[0]}')

        plt.margins(0.01,0.01)
        plt.title('fuel intensities for chosen cohorts')
        plt.xlabel('Year (time)') 
        plt.ylabel('fuel intensity')
        plt.legend(loc = "lower left")
        plt.show()
        return