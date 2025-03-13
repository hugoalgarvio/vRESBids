## Old Market Design - Stage 1 - Perfect Case for All Markets
import numpy as np
import pandas as pd

# Load data from the Excel file
file_path =df = 'Market-Prices2.xlsx'
df = pd.read_excel(file_path)

# Parameters

## Market Data
dam_price = df.iloc[:, 0].values ## Price in DAM
exc_price = df.iloc[:, 23].values ## Price for Positive Imbalance (Excess Energy)
def_price = df.iloc[:, 24].values ## Price for Negative Imbalance
ida_price = df.iloc[:, 25].values ### Price in IDA

trup_price = df.iloc[:, 26].values ## Price for Upward TR 
trdown_price  = df.iloc[:, 28].values ## Price for Downward TR
trup_req = df.iloc[:, 27].values ## TSO requirements for Upward TR 
trdown_req = df.iloc[:, 29].values ## TSO requirements for Downward TR

afrr_price = df.iloc[:, 30].values ## Price for SR capacity
afrr_nup = df.iloc[:, 31].values ## TSO needs for Upward SR capacity 
afrr_ndown = df.iloc[:, 32].values ## TSO needs for Downward SR capacity
afrr_upused = df.iloc[:, 33].values ## Upward SR energy used
afrr_downused = df.iloc[:, 34].values ## Downward SR energy used

deficit = df.iloc[:,38].values  ## Negative Imbalance Price for BSP
excess = df.iloc[:,39].values  ## Positive Imbalance Price for BSP


## Constants
min_bid = 0.1  ### Minimum bid size
cap_nom = 250  ### Nominal Capacity of Wind Power Plant

### Wind Data
p_obs = df.iloc[:, 1].values ## Observed Power
quantiles = df.iloc[:, 2:23].values ## Probabilistic Quantiles
quantiles_header = df.columns[2:23] 
ida_forecast = df.iloc[:, 35].values ## Deterministic Forecast in IDA
idc_forecast = df.iloc[:, 36].values ## Deterministic Forecast 15min before (IDC)

# Decision Variables

optimal_quantiles = np.zeros_like(p_obs, dtype=float) ## Optimal bid in DAM for each time step
optimal_quantiles_header = np.empty(len(p_obs), dtype=object)  

exc_sell = np.zeros_like(p_obs, dtype=float) ## Excess energy sold in Imbalance Settlement
def_buy = np.zeros_like(p_obs, dtype=float) ## Negative Imbalance in IS

ida_sell = np.zeros_like(p_obs, dtype=float) ## Volume sold in IDA
ida_sell1 = np.zeros_like(p_obs, dtype=float) ## Volume sold in IDA when also participatin in reserve

ida_buy = np.zeros_like(p_obs, dtype=float) ## Volume bought in IDA
ida_buy1 = np.zeros_like(p_obs, dtype=float) ## Volume bought in IDA when also participatin in reserve

idc_sell = np.zeros_like(p_obs, dtype=float) # Volume sold in IDC
idc_sell1 = np.zeros_like(p_obs, dtype=float) # Volume sold in IDC when also participatin in reserve
idc_buy = np.zeros_like(p_obs, dtype=float) # Volume bought in IDC 
idc_buy1 = np.zeros_like(p_obs, dtype=float) # Volume bought in IDC when also participatin in reserve


sr_upcap = np.zeros_like(p_obs, dtype=float) ## Allocated Upward SR capacity
sr_downcap = np.zeros_like(p_obs, dtype=float) ## Allocated Downward SR capacity

sr_up = np.zeros_like(p_obs, dtype=float) ## Up SR activated
sr_down = np.zeros_like(p_obs, dtype=float) ## Down SR activated

tr_up = np.zeros_like(p_obs, dtype=float) ## Upward TR energy activated
tr_down = np.zeros_like(p_obs, dtype=float) ## Downward TR energy activated

curtail = np.zeros_like(p_obs, dtype=float) ## Energy Curtailed
exc_sell1 = np.zeros_like(p_obs, dtype=float) ## Positive Imbalance in IS for BSP
def_buy1 = np.zeros_like(p_obs, dtype=float) ## Negative Imbalance in IS for BSP

total_revenue = np.zeros_like(p_obs, dtype=float) ## Total Revenue 
#%% Participation only in DAM

def calculate_dam(deviation: np.ndarray, quantiles: np.ndarray, dam_price: float, exc_price: float, def_price: float) -> tuple:
   
    dam_revenue = quantiles * dam_price  # Calculate DAM revenue for all quantiles
    
    exc_sell = np.where(deviation > 0, deviation, 0)  # If deviation is positive, sell excess
    def_buy = np.where(deviation < 0, -deviation, 0)  # If deviation is negative, buy deficit
    
    # Revenue from excess and deficit
    exc_revenue = exc_sell * exc_price
    def_revenue = def_buy * def_price
    
    total_revenue = dam_revenue + exc_revenue - def_revenue  # Combine DAM and deviation revenues
    
    return total_revenue, exc_sell, def_buy


# Iteration through time steps: Calculate optimal bids for each time step based on the revenues
for i in range(len(p_obs)):
    
    deviation = p_obs[i] - quantiles[i]  
    
    #Calculate revenue for each quantile by evaluating the action and market prices defined in function
    revenues = np.array([
        calculate_dam(d, q, dam_price[i], exc_price[i], def_price[i])
        for d, q in zip(deviation, quantiles[i])
    ])

    optimal_index = np.argmax(revenues[:, 0]) # Select the quantile that maximizes revenue
    
    # Store the optimal quantile for this time step
    optimal_quantiles[i] = quantiles[i, optimal_index]
    
    # Store the revenue and the corresponding actions for the optimal quantile
    total_revenue[i], exc_sell[i], def_buy[i] = revenues[optimal_index]

# Average Remuneration
print(f'Avg Remuneration DAM: {np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')

#%% Participation in DAM and IDM

def calculate_idm (dev: float, quantiles: float, dam_price: float, exc_price: float, def_price: float, ida_price: float) -> tuple:
    
    exc_sell, def_buy, ida_sell, ida_buy = 0, 0, 0, 0  # Initialize all outputs
    
    dam_revenue = quantiles * dam_price

    if dev > 0: ## Positive Imbalance
        if ida_price > exc_price:   
            ida_sell = dev
        else:
            exc_sell = dev
        total_revenue = dam_revenue + (dev * max(ida_price, exc_price))  # Revenue based on best price
    
    else: ## Negative Imbalance
        if ida_price < def_price:
            ida_buy = abs(dev)
        else:
            def_buy = abs(dev)
        total_revenue = dam_revenue - (abs(dev) * min(ida_price, def_price)) 

    return total_revenue, exc_sell, def_buy, ida_sell, ida_buy


# Iteration through time steps: Calculate optimal bids for each time step based on the revenues
for i in range(len(p_obs)):
    dev = p_obs[i] - quantiles[i]
    
    #Calculate revenue for each quantile by evaluating the action and market prices defined
    revenues = np.array([
        calculate_idm(d, q, dam_price[i], exc_price[i], def_price[i], ida_price[i])
        for d, q in zip(dev, quantiles[i])
    ])
    
    optimal_index = np.argmax(revenues[:, 0]) # Select the quantile that maximizes revenue
    
    # Store the revenue and the corresponding actions for the optimal quantile
    optimal_quantiles[i], total_revenue[i], exc_sell[i], \
    def_buy[i], ida_sell[i], ida_buy[i] = quantiles[i, optimal_index], *revenues[optimal_index]

# Avera Remuneration
print(f'Avg Remuneration DAM+IDM: {np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')

#%% Participation in SR 
def calculate_sr(deviation,quantiles,p_obs,dam_price, ida_price,exc_price,def_price, afrr_price, afrr_nup, afrr_ndown,afrr_upused, afrr_downused, trup_price, trdown_price,excess,deficit):
  
    sr_upcap, sr_downcap, sr_up, sr_down, sr_downcap_others = 0,0,0,0,0
    exc_sell1, ida_sell1, exc_sell, ida_sell = 0,0,0,0
    def_buy1, ida_buy1, def_buy, ida_buy, curtail = 0,0,0,0,0
    down_revenue, up_revenue, up_en, down_en  = 0,0,0,0
    curt_up, curt_down = 0,0
    ida_up, ida_down = 0,0
    exc_up, exc_down = 0,0
    def_up, def_down = 0,0
    
    dam_rem = quantiles * dam_price
    
    if deviation > 0:
        
        ida_sell_rem = deviation * ida_price
        dev_rem = deviation * exc_price
        
        if afrr_nup > min_bid and deviation > min_bid: ## Participation in SR is possible
            ###################################################################            
            if afrr_upused > 0 and afrr_downused == 0:
                # Capacity Market
                sr_upcap = min(afrr_nup, deviation)
                sr_downcap = min(afrr_ndown, sr_upcap/2)
                cap_rem = (sr_upcap * afrr_price) + (sr_downcap * afrr_price)
                # Activation of Energy Bids
                sr_up = min(afrr_upused, sr_upcap)
                en_rem = sr_up * trup_price
                remaining_dev = deviation - sr_up
                ### Sell remaining deviation to highest price 
                if ida_price > excess:
                    ## Quantity to sell in IDA limited by available capacity
                    position = quantiles + sr_upcap
                    available_capacity = cap_nom - position
                    ida_sell1 = min(available_capacity, remaining_dev)
                    ida_rem = ida_sell1 * ida_price
                    # Imbalance Settlement
                    final_schedule = quantiles + sr_up + ida_sell1
                    final_deviation = p_obs - final_schedule
                    if excess > 0:
                        exc_sell1 = final_deviation
                        exc_rem = exc_sell1 * excess
                        sr_rem = cap_rem + en_rem + ida_rem + exc_rem
                    else:
                        curtail = final_deviation
                        sr_rem = cap_rem + en_rem + ida_rem
                else:
                    final_schedule = quantiles + sr_up
                    final_deviation = p_obs - final_schedule
                    exc_sell1 = final_deviation
                    exc_rem = exc_sell1 * excess
                    sr_rem = cap_rem + en_rem + exc_rem
                    
            ######################################################################                    
            elif afrr_upused == 0 and afrr_downused > 0:
                # Capacity Market
                sr_upcap = min(afrr_nup, deviation)
                sr_downcap = min(afrr_ndown, sr_upcap/2)
                cap_rem = (sr_upcap*afrr_price) + (sr_downcap * afrr_price)
                sr_downcap_others =  afrr_ndown - sr_downcap # Down needs covered by other producers 
                ### Downward activation required from WPP
                down_req = min(max(0,afrr_downused- sr_downcap_others), sr_downcap)
                
                ### If the required activation is higher than scheduled injection
                if down_req > quantiles:
                    ## Sell more quantity in IDA to increase position/schedule
                    ida_sell1 = min(cap_nom - quantiles - sr_upcap, down_req - quantiles)
                    ida_rem = ida_sell1 * ida_price
                    if ida_sell1 + quantiles < down_req:# If still not possible to comply with activation of downward reserve 
                        ### Limit capacity offer to the schedule
                        sr_downcap = min(afrr_ndown, deviation/2, quantiles)
                        sr_upcap = min(sr_downcap*2, deviation, afrr_nup)
                        cap_rem = (sr_upcap * afrr_price) + (sr_downcap * afrr_price)
                        down_req = min(max(0,afrr_downused- sr_downcap_others), sr_downcap)
                        sr_down = down_req
                        en_rem = - down_req * trdown_price
                        final_schedule = quantiles - down_req
                        if ida_price > excess:
                            remaining_capacity = cap_nom - quantiles - sr_upcap 
                            ida_sell1 = min(remaining_capacity, final_schedule)
                            ida_rem = ida_sell1 * ida_price
                            final_schedule = quantiles + ida_sell1 - sr_down
                            final_deviation = p_obs - final_schedule
                            if excess > 0:
                                final_deviation = p_obs - final_schedule
                                exc_sell1 = final_deviation
                                exc_rem = exc_sell1 * excess
                                sr_rem = cap_rem + ida_rem + en_rem + exc_rem
                            else:
                                curtail = final_deviation
                                sr_rem = cap_rem + ida_rem +  en_rem
                        else:
                            final_deviation = p_obs - final_schedule
                            exc_sell1 = final_deviation
                            exc_rem = exc_sell1 * excess
                            sr_rem = cap_rem + en_rem + exc_rem
           
                    else: ## After making adjustments in IDA. 
                        sr_down = down_req
                        en_rem = -sr_down * trdown_price
                        final_schedule = quantiles + ida_sell1 - sr_down 
                        final_deviation = p_obs - final_schedule
                        if excess > 0:
                            exc_sell1 = final_deviation
                            exc_rem = exc_sell1 * excess
                            sr_rem = cap_rem + en_rem + ida_rem + exc_rem
                       
                        else: ##for negative prices, remaining energy is curtailed
                            curtail = final_deviation
                            sr_rem = cap_rem + en_rem + ida_rem
              
                else: ## Activation of initial offer
                    sr_down = down_req
                    en_rem = -sr_down * trdown_price
                    if ida_price > excess: # Adjustments in IDA
                        ida_sell1 = min(cap_nom - quantiles - sr_upcap, deviation - sr_down)
                        ida_rem = ida_sell1 * ida_price
                        final_schedule = quantiles + ida_sell1 - sr_down
                        final_deviation = p_obs - final_schedule
                        if excess > 0:
                            exc_sell1 = final_deviation
                            exc_rem = exc_sell1 * excess
                            sr_rem = cap_rem + en_rem + ida_rem + exc_rem
                        else:
                            curtail = final_deviation
                            sr_rem = cap_rem + en_rem + ida_rem
                    else:
                        final_schedule = quantiles - sr_down
                        final_deviation = p_obs - final_schedule
                        exc_sell1 = final_deviation
                        exc_rem = exc_sell1 * excess
                        sr_rem = cap_rem + en_rem + exc_rem
                   
            ###################################################################
            elif afrr_upused > 0 and afrr_downused > 0:
                ### Energy used in both direction in the same hour. Check which direction yields higher revenue
                
                ## Capacity Market
                sr_upcap = min(afrr_nup, deviation)
                sr_downcap = min(afrr_ndown, sr_upcap/2)
                cap_rem = (sr_upcap * afrr_price) + (sr_downcap * afrr_price)
                sr_downcap_others =  afrr_ndown - sr_downcap
                down_req = min(max(0,afrr_downused- sr_downcap_others), sr_downcap)
            
                ## Option 1: Upward Regulation 
                up_en = min(afrr_upused, sr_upcap) 
                en_up = up_en * trup_price 
                if ida_price > excess: # Make adjustments in IDA
                    available_capacity = cap_nom - quantiles - sr_upcap
                    remaining_deviation = p_obs - quantiles - up_en
                    ida_up = min(available_capacity, remaining_deviation)
                    ida_rem = ida_up * ida_price
                else:
                    ida_up, ida_rem = 0,0
                ## Imbalance Settlement
                final_schedule = quantiles + ida_up + up_en
                final_deviation_up = p_obs - final_schedule
                if excess > 0:
                    exc_up = final_deviation_up
                    exc_rem = exc_up * excess
                    up_revenue = cap_rem + en_up + ida_rem + exc_rem
                else:
                    curt_up = final_deviation_up
                    up_revenue = cap_rem + en_up + ida_rem
              
                ## Option 2: Downward Regulation
                down_en = down_req
                en_down = - down_en * trdown_price
                if ida_price > excess: # Make adjustments in IDA
                    available_capacity = cap_nom - quantiles - sr_upcap
                    remaining_deviation = p_obs - quantiles + down_en
                    ida_down = min(available_capacity, remaining_deviation)
                    ida_rem = ida_down * ida_price
                else:
                    ida_down, ida_rem = 0,0
                ## Imbalance Settlement
                final_schedule = quantiles + ida_down - down_en
                final_deviation_down = p_obs - final_schedule
                if excess > 0:
                    exc_down = final_deviation_down
                    exc_rem = exc_down * excess
                    down_revenue = cap_rem + en_down + ida_rem + exc_rem
                else:
                    curt_down = final_deviation_down
                    down_revenue = cap_rem + en_down + ida_rem
      
                ## Choose if Upward or Downward  
                if up_revenue > down_revenue or down_req <= 0 or down_req > quantiles:
                    ida_sell1 = ida_up
                    curtail = curt_up
                    exc_sell1 = exc_up
                    sr_up = up_en
                    sr_rem = up_revenue
                else:
                    ida_sell1 = ida_down
                    curtail = curt_down
                    exc_sell1 = exc_down
                    sr_down = down_en
                    sr_rem = down_revenue
                    
###############################################################################
            else: ## No SR energy used 
                sr_upcap = min(afrr_nup, cap_nom - quantiles) 
                sr_downcap = min(afrr_ndown, sr_upcap/2)
                cap_rem = (sr_upcap * afrr_price) + (sr_downcap * afrr_price)
                if ida_price > excess:
                    available_capacity = cap_nom - quantiles - sr_upcap
                    ida_sell1 = min(available_capacity, deviation)
                    ida_rem = ida_sell1 * ida_price
                    final_schedule = quantiles + ida_sell1
                    final_deviation = p_obs - final_schedule
                    if excess > 0:
                        exc_sell1 = final_deviation
                        exc_rem = exc_sell1 * excess
                        sr_rem = cap_rem + ida_rem +  exc_rem      
                    else:
                        curtail = final_deviation
                        sr_rem = cap_rem +  ida_rem
                else:
                    exc_sell1 = deviation
                    exc_rem = exc_sell1 * excess
                    sr_rem = cap_rem +  exc_rem
        else: ## No reserve in this hour
            sr_rem = 0
       
        total_revenue_ida = dam_rem + ida_sell_rem
        total_revenue_dev = dam_rem + dev_rem
        total_revenue_mix = dam_rem + sr_rem
            
        if total_revenue_ida >= total_revenue_dev and total_revenue_ida >= total_revenue_mix:
            ida_sell1, exc_sell1, exc_sell, sr_up, sr_down, sr_upcap, sr_downcap, curtail = 0,0,0,0,0,0,0,0
            ida_sell = abs(deviation)
        elif (total_revenue_dev > total_revenue_ida) and (total_revenue_dev > total_revenue_mix):
            ida_sell1, exc_sell1, ida_sell, sr_up, sr_down, sr_upcap, sr_downcap, curtail = 0,0,0,0,0,0,0,0
            exc_sell = deviation
            
    else: ### Deficit
        
        ida_buy_rem = deviation * ida_price
        dev_rem = deviation * def_price
        
        if afrr_ndown > min_bid and abs(deviation) > min_bid :
            if afrr_upused > 0 and afrr_downused == 0:
                # Capacity Market
                sr_upcap = min(cap_nom - quantiles, afrr_nup)
                sr_downcap = min(sr_upcap/2, afrr_ndown)
                cap_rem = (sr_upcap * afrr_price) + (sr_downcap * afrr_price)
                
                # Real-time
                up_req = min(afrr_upused, sr_upcap) ## TSO's request for UP regulation
                final_schedule = quantiles + up_req
                final_deviation = p_obs - final_schedule
                
                # Imbalance Settlement
                def_buy1 = abs(final_deviation)
                def_rem = - def_buy1 * deficit
                sr_rem = cap_rem + def_rem
            
            elif afrr_upused == 0 and afrr_downused > 0: 
                # Capacity Market
                sr_downcap = min(quantiles, afrr_ndown, (cap_nom-quantiles)/2)
                sr_upcap = min(sr_downcap*2, afrr_nup, cap_nom - quantiles)
                cap_rem = (sr_upcap*afrr_price) + (sr_downcap * afrr_price)
                sr_downcap_others =  afrr_ndown - sr_downcap
                
                # Real-time
                down_req = min(sr_downcap, max(0, afrr_downused - sr_downcap_others))
                sr_down = down_req
                en_down_rem = - sr_down * trdown_price
                
                # Imbalance Settlement
                final_schedule = quantiles - sr_down
                final_deviation = p_obs - final_schedule
                if final_deviation > 0:
                    if excess > 0:
                        exc_sell1 = final_deviation
                        exc_rem = exc_sell1 * excess
                        sr_rem = cap_rem + en_down_rem + exc_rem
                    else:
                        curtail = final_deviation
                        sr_rem = cap_rem + en_down_rem 
                else:
                    def_buy1 = abs(final_deviation)
                    def_rem = - def_buy1 * deficit
                    sr_rem = cap_rem + en_down_rem + def_rem
                
            elif afrr_upused == 0 and afrr_downused == 0:
                sr_upcap = min(afrr_nup, cap_nom - quantiles)
                sr_downcap = min(afrr_ndown, sr_upcap/2)
                cap_rem = (sr_upcap*afrr_price) + (sr_downcap * afrr_price)
                if deficit < ida_price:
                    def_buy1 = abs(deviation)
                    def_rem = - def_buy1 * deficit
                    sr_rem = cap_rem + def_rem
                else:
                    ida_buy1 = abs(deviation)
                    ida_rem = - ida_buy1 * ida_price
                    sr_rem = cap_rem + ida_rem
                    
            else: ## Energy used in both directions. Check which direction yields higher revenue
                ### Capacity Market
                sr_downcap = min(quantiles, afrr_ndown, (cap_nom-quantiles)/2)
                sr_upcap = min(sr_downcap*2, afrr_nup, cap_nom - quantiles)
                cap_rem = (sr_upcap * afrr_price) + (sr_downcap * afrr_price)
                sr_downcap_others =  afrr_ndown - sr_downcap
                down_req = min(max(0,afrr_downused- sr_downcap_others), sr_downcap)
            
                # Option 1: Upward Regulation ################################
                up_req = min(afrr_upused, sr_upcap) ## TSO's request for UP regulation
                final_schedule = quantiles + up_req
                final_up = p_obs - final_schedule
                def_up = abs(final_up)
                def_rem = - def_up * deficit
                up_revenue = cap_rem + def_rem
                    
                # Option 2: Downward Regulation ##############################
                down_en = down_req
                en_down = - down_en * trdown_price
                final_schedule = quantiles - down_en
                final_imbalance = p_obs - final_schedule
                if final_imbalance > 0:
                    if excess > 0:
                        exc_sell1 = final_imbalance
                        exc_rem = exc_sell1 * excess
                        down_revenue = cap_rem + en_down  + exc_rem
                    else:
                        curtail = final_imbalance
                        down_revenue = cap_rem + en_down 
                else:
                    def_down =  abs(final_imbalance)
                    def_rem = - def_down * deficit
                    down_revenue = cap_rem + en_down +  def_rem
                ## Choose Upward or Downward   
                if up_revenue > down_revenue or down_req <= 0: ## Upward Regulation
                    def_buy1 = def_up
                    sr_up = up_en
                    sr_rem = up_revenue
                else:
                    def_buy1 = def_down
                    sr_down = down_en
                    sr_rem = down_revenue
        else: ## No reserve needs.
            sr_rem = float('-inf')
            
        total_revenue_ida = dam_rem + ida_buy_rem
        total_revenue_dev = dam_rem + dev_rem
        total_revenue_mix = dam_rem + sr_rem
            
        if (total_revenue_ida >= total_revenue_dev) and (total_revenue_ida >= total_revenue_mix):
            ida_buy1, def_buy1, def_buy, sr_down, sr_upcap, sr_up,sr_downcap,curtail = 0,0,0,0,0,0,0,0
            ida_buy = abs(deviation)
        elif (total_revenue_dev > total_revenue_ida) and (total_revenue_dev > total_revenue_mix):
            ida_buy1, def_buy1, ida_buy, sr_down, sr_upcap,sr_up, sr_downcap,curtail = 0,0,0,0,0,0,0,0
            def_buy = abs(deviation)

    total_revenue = max(total_revenue_ida, total_revenue_dev, total_revenue_mix)
        
    return (total_revenue, sr_upcap, sr_downcap, sr_down, sr_up, def_buy1, ida_buy1, ida_buy, def_buy, curtail, exc_sell1, ida_sell1, ida_sell, exc_sell)
            
# Iteration through time steps: Calculate optimal bids for each time step based on the revenues
for i in range(len(p_obs)):
    
    # Calculate deviation for each quantile 
    deviation = p_obs[i] - quantiles[i]

    #Calculate revenue for each quantile by evaluating the action and market prices defined in function
    revenues = np.array([
        calculate_sr(d, q, p_obs[i],dam_price[i], ida_price[i],exc_price[i],def_price[i],\
                             afrr_price[i], afrr_nup[i], afrr_ndown[i],afrr_upused[i], afrr_downused[i],\
                             trup_price[i], trdown_price[i],excess[i],deficit[i])
        for d, q in zip(deviation, quantiles[i])
    ])

    optimal_index = np.argmax(revenues[:, 0]) # Select the quantile that maximizes revenue
    optimal_quantiles[i] = quantiles[i, optimal_index] # Store the optimal quantile for this time step
    optimal_quantiles_header[i] = quantiles_header[optimal_index] # Store the header for the optimal quantile
    
    # Store the revenue and the corresponding actions for the optimal quantile
    total_revenue[i], sr_upcap[i], sr_downcap[i], sr_down[i], sr_up[i], def_buy1[i],\
    ida_buy1[i], ida_buy[i], def_buy[i], curtail[i], exc_sell1[i], ida_sell1[i],\
    ida_sell[i], exc_sell[i] = revenues[optimal_index]

# Average Remuneration
print(f'Avg Remuneration SR: {np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')
#%% Participation in Tertiary Reserve

# Function to calculate revenue and actions
def calculate_tr( deviation, quantiles, dam_price, ida_price, exc_price, def_price, 
                        trup_req, trup_price, trdown_req, trdown_price, excess, deficit) -> tuple:

    
    tr_up, tr_down, ida_sell1, ida_buy1, exc_sell1, def_buy1, ida_sell, ida_buy, exc_sell, def_buy = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    dam_rem = quantiles * dam_price ##Revenue from DAM for each quantile 
    
    
    if deviation > 0: ###Positive Deviation
        
        ##Potential revenues from selling excess in IDA or IS
        
        ida_rem = deviation * ida_price
        dev_rem = deviation * exc_price

        ### Participation in Upward TR 
        
        if trup_req > 0 and deviation >= min_bid: 
        ## Only if TSO requires TR energy and the deviation is higher than min bid size
            
            if excess > trup_price: ## If the price for excess is higher than the energy price
                
                tr_up = min(min_bid, deviation, trup_req) ## Sell min bid size to participate in reserve
                trup_rem = tr_up * trup_price ## Revenue from Upward regulation
                
                remaining_dev = max(0, deviation - tr_up) ## Remaining deviation
                exc_sell1 = remaining_dev
                mix_rem = trup_rem + (exc_sell1 * excess)
           
            else: ## Sell all the excess to TR
                
                tr_up = min(deviation, trup_req)
                trup_rem = tr_up * trup_price
                remaining_dev = max(0, deviation - tr_up)
                
                if excess > ida_price:
                    exc_sell1 = remaining_dev
                    mix_rem = trup_rem + (exc_sell1 * excess)
                else:
                    ida_sell1 = remaining_dev
                    mix_rem = trup_rem + (ida_sell1 * ida_price)
                    
        else: ## There is no reserve in this hour
            mix_rem = 0
            
            
        ### Calculate total revenue from different markets

        total_revenue_ida = dam_rem + ida_rem
        total_revenue_dev = dam_rem + dev_rem
        total_revenue_mix = dam_rem + mix_rem
        
        ### Store quantities when not participating in TR 
        
        if total_revenue_ida == max(total_revenue_ida ,total_revenue_dev, total_revenue_mix):
            ida_sell1, exc_sell1, exc_sell, tr_up = 0,0,0,0
            ida_sell = deviation
        elif total_revenue_dev == max(total_revenue_ida ,total_revenue_dev, total_revenue_mix):
            ida_sell, ida_sell1, exc_sell1, tr_up = 0,0,0,0
            exc_sell = deviation

    else: ### Negative Deviation
        
        ida_rem = deviation * ida_price
        dev_rem = deviation * def_price
     
        if trdown_req > 0 and abs(deviation) >= min_bid:
            if deficit < trdown_price:
                tr_down = min(abs(deviation), trdown_req, min_bid)
                trdown_rem = -tr_down * trdown_price
                remaining_dev = max(0, abs(deviation) - tr_down)
                def_buy1 = remaining_dev
                mix_rem = trdown_rem - def_buy1 * deficit
            else:
                tr_down = min(abs(deviation), trdown_req)
                trdown_rem = -tr_down * trdown_price
                remaining_dev = max(0, abs(deviation) - tr_down)
               
                if ida_price < deficit:
                    ida_buy1 = remaining_dev
                    mix_rem = trdown_rem - ida_buy1 * ida_price
                else:
                    def_buy1 = remaining_dev
                    mix_rem = trdown_rem - def_buy1 * deficit
        else:
            mix_rem = float('-inf')

        total_revenue_ida = dam_rem + ida_rem
        total_revenue_dev = dam_rem + dev_rem
        total_revenue_mix = dam_rem + mix_rem

        if total_revenue_ida == max(total_revenue_ida ,total_revenue_dev, total_revenue_mix):
            tr_down, def_buy1, ida_buy1, def_buy = 0,0,0,0
            ida_buy = abs(deviation)
        elif total_revenue_dev == max(total_revenue_ida ,total_revenue_dev, total_revenue_mix):
            tr_down, def_buy1, ida_buy1, ida_buy = 0,0,0,0
            def_buy = abs(deviation)

    
    ## Choose the strategy that maximizes revenue
    total_revenue = max(total_revenue_ida, total_revenue_dev, total_revenue_mix)

    return (total_revenue, tr_up, tr_down, exc_sell, def_buy, ida_sell, ida_buy, exc_sell1, def_buy1, ida_sell1, ida_buy1)


# Iteration through time steps: Calculate optimal bids for each time step based on the revenues

for i in range(len(p_obs)):
    
    # Calculate deviation for each quantile 
    deviation = p_obs[i] - quantiles[i]
    
    #Calculate revenue for each quantile by evaluating the action and market prices defined in function
    revenues = np.array([
        calculate_tr(d, q, dam_price[i], ida_price[i], exc_price[i], def_price[i], 
                            trup_req[i], trup_price[i], trdown_req[i], trdown_price[i], 
                            excess[i], deficit[i])
        for d, q in zip(deviation, quantiles[i])
    ])
    
    optimal_index = np.argmax(revenues[:, 0]) # Select the quantile that maximizes revenue
    
    # Store the optimal quantile for this time step
    optimal_quantiles[i] = quantiles[i, optimal_index] 
    
    # Store the revenue and the corresponding actions for the optimal quantile
    total_revenue[i], tr_up[i], tr_down[i], exc_sell[i], def_buy[i], ida_sell[i], ida_buy[i], exc_sell1[i], def_buy1[i], ida_sell1[i], ida_buy1[i] = revenues[optimal_index]

# Average Remuneration
print(f'Avg Remuneration TR: {np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')
