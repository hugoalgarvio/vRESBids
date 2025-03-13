## Old Market Design - Stage 2 - Operational Case
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
#%% Bid in DAM
threshold = 41.63
step_size = threshold / 21  # To create 21 values between 0 and threshold
price_int = np.linspace(0, threshold, 21)  # Price interval

# Find the index of the closest bid value for each DAM price
idx_closest_price = np.searchsorted(price_int, dam_price, side='right') - 1

# Initialize a list to store quantile headers for each observation
optimal_quantiles_header = []

    # Select the appropriate quantile for each DAM price
for i in range(len(dam_price)):
    if dam_price[i] >= threshold:
        optimal_quantiles = quantiles[i][-1]
        optimal_quantiles_header.append('Q0.98')  # If DAM price >= threshold, bid 'Q0.98'
    else:
        closest_price = price_int[idx_closest_price[i]]  # Closest bid value
        closest_idx = np.where(price_int == closest_price)[0][0]  # Index of the closest bid
        optimal_quantiles_header.append(quantiles_header[closest_idx])  # Header for closest quantile

    # Convert optimal_quantiles to a NumPy array
optimal_quantiles = np.where(dam_price >= threshold,
                                 quantiles[:, -1],
                                 quantiles[np.arange(len(dam_price)), idx_closest_price])

#%% Participation in Secondary Reserve
## Supporting functions for SR participation

def calculate_en_activation (up_req, down_req, afrr_upused, afrr_downused, avcap, prog,
                p_obs, trup_price, trdown_price, deficit, excess):
    down_sr, up_sr  = 0,0
    en_up, en_down =0,0 
    en_rem, final_deviation = 0,0
    expected_rev_up, expected_rev_down = 0,0
    
    ### Activation of energy bids
    if afrr_upused > 0 and afrr_downused <= 0: # Only UP
        up_sr = max(0,min(up_req, avcap))
        en_rem = up_sr * trup_price
        missing_up = up_req - up_sr
        final_deviation = p_obs - prog- up_sr - missing_up
    elif afrr_upused <= 0 and afrr_downused > 0: #Only DOWN
        down_sr = down_req
        en_rem = - down_sr * trdown_price
        final_schedule = prog - down_sr
        final_deviation = p_obs - final_schedule
    elif afrr_upused <= 0 and afrr_downused <= 0: # No reserve used
        en_rem = 0
        final_schedule = prog
        final_deviation = p_obs - final_schedule
    else: ## Requested in both directions
        if down_req <= 0:
            up_sr = max(0,min(up_req, avcap))
            en_rem = up_sr * trup_price
            missing_up = up_req - up_sr
            final_deviation = p_obs - prog - up_sr - missing_up
                    
        else: 
            ## Option 1: UP
            en_up = max(0,min(up_req, avcap))
            en_rev_up = en_up * trup_price
            up_en_missing = up_req - en_up
            rev_dev_up = up_en_missing * deficit
            expected_rev_up = en_rev_up + rev_dev_up
                        
            ##Option 2: DOWN
            en_down = down_req
            en_rev_down = - down_req * trdown_price
            final_schedule_down = prog - down_req
            final_dev_down = p_obs - final_schedule_down
             
            if final_dev_down > 0:
                rev_dev_down = np.where(excess > 0, final_dev_down * excess, 0)
            else:
                rev_dev_down = final_dev_down * deficit
            
            expected_rev_down = en_rev_down + rev_dev_down
                        
            if (expected_rev_down > expected_rev_up):
                ## DOWN
                en_rem = en_rev_down
                down_sr = en_down
                final_deviation = p_obs - prog + en_down
                
            else:
                ## UP
                final_deviation = p_obs - prog - en_up - up_en_missing
                en_rem = en_rev_up
                up_sr = en_up
                        
    return final_deviation, up_sr, down_sr, en_rem

def revenue_IS (imbalance, exc_price, def_price):
    def_b, exc_s, imb_rem = 0,0,0
    if imbalance > 0:
        exc_s = imbalance
        imb_rem = exc_s * exc_price
    else:
        def_b = abs(imbalance)
        imb_rem = - def_b * def_price
    return exc_s, def_b, imb_rem

def revenue_IS_SR (final_deviation, excess, deficit):
    exc_s1, def_b1, dev_rem, curt = 0,0,0,0
    
    if final_deviation > 0:
        if excess > 0:
            exc_s1 = final_deviation
            dev_rem = exc_s1 * excess
        else:
            curt = final_deviation
            dev_rem = 0
    else:
        def_b1 = abs(final_deviation)
        dev_rem = - def_b1 * deficit
    return exc_s1, def_b1, dev_rem, curt

def calculate_cap (prog, afrr_ndown, afrr_nup, deviation, afrr_price):
    down_cap = min(prog, deviation/2, afrr_ndown)
    up_cap = min(deviation, down_cap*2, afrr_nup)
    cap_rem = (down_cap * afrr_price) + (up_cap * afrr_price)
    down_co = afrr_ndown - down_cap
    
    return down_cap, up_cap, down_co, cap_rem

#%% Strategy 1
def calculate_S1 (optimal_quantiles,p_obs,dam_price, ida_price,exc_price,def_price, afrr_price, afrr_nup, afrr_ndown,afrr_upused, afrr_downused, trup_price, trdown_price,excess,deficit,\
                         ida_forecast):
  
    ida_s, ida_s1, ida_b, ida_b1 = 0,0,0,0
    exc_s, exc_s1, def_b, def_b1 = 0,0,0,0
    down_cap, up_cap, down_sr, up_sr, curt = 0,0,0,0,0
    down_req, up_req = 0,0
  
    dam_rem = optimal_quantiles * dam_price
    
    dev = ida_forecast - optimal_quantiles

    if dev < 0: ## Negative deviation in IDA
        ida_b = 1.05 * abs(dev)
        ida_rem = - ida_b * ida_price
        # Imbalance Settlement
        imbalance = p_obs - optimal_quantiles + ida_b
        exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
        total_rev = dam_rem + ida_rem + imb_rem
       

    else: # Positive deviation hours before delivery

        if afrr_nup > min_bid and optimal_quantiles > min_bid and dev > min_bid:
          
            ## Capacity Market
            prog = optimal_quantiles
            deviation = dev
            down_cap, up_cap, down_co, cap_rem = calculate_cap (prog, afrr_ndown, afrr_nup, deviation, afrr_price)
            
            # IDA
            if up_cap < dev:
                ida_s1 = max(0, dev - up_cap)
                ida_rem = ida_s1 * ida_price
            else:
                ida_s1 = 0
                ida_rem = 0
                  
            prog = optimal_quantiles + ida_s1 ## Final position before activation of bids
            
            ### Real-time
            avcap = p_obs - prog # Available capacity after complying with dispatch
                
            ### TSO requests for activation
            up_req = min(afrr_upused, up_cap)
            down_req = min(max(0,afrr_downused - down_co), down_cap)
                
            ### Activation of energy bids
            final_deviation, up_sr, down_sr, en_rem = calculate_en_activation(
                up_req, down_req, afrr_upused, afrr_downused, avcap, prog,
                p_obs, trup_price, trdown_price, deficit, excess)
                        
            # Imbalance Settlement      
            exc_s1, def_b1, dev_rem, curt = revenue_IS_SR (final_deviation, excess, deficit)
            total_rev = dam_rem + cap_rem + ida_rem + en_rem + dev_rem
            
        else: # No participation in Reserve. Sell expected surplus in IDA.
            
            ida_s = 0.95 *dev ## More conservative approach
            ida_rem = ida_s * ida_price
            # Imbalance Settlement
            imbalance = p_obs - optimal_quantiles - ida_s
            exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
            total_rev = dam_rem + ida_rem + imb_rem
                
            
    return total_rev, ida_s, ida_s1, ida_b, ida_b1, exc_s, exc_s1, def_b, def_b1, down_cap, up_cap, down_sr, up_sr, curt

# Iterate through each time step 
for i in range(len(p_obs)):
    
    # Call the function for each time step and unpack the results
    total_rev, ida_s, ida_s1, ida_b, ida_b1, exc_s, exc_s1, def_b, def_b1, down_cap,\
    up_cap, down_sr, up_sr, curt = calculate_S1(optimal_quantiles[i],
        p_obs[i], dam_price[i] ,ida_price[i], exc_price[i], def_price[i], afrr_price[i],
        afrr_nup[i], afrr_ndown[i],afrr_upused[i],afrr_downused[i],trup_price[i],
        trdown_price[i] ,excess[i],deficit[i],ida_forecast[i],
        
    )
    # Store the total revenue for the each time step
    total_revenue[i] = total_rev
    ida_sell[i] = ida_s
    ida_sell1[i] = ida_s1
    exc_sell[i] = exc_s
    exc_sell1[i] = exc_s1
    ida_buy[i] = ida_b
    ida_buy1[i] = ida_b1
    def_buy[i] = def_b
    def_buy1[i] = def_b1
    sr_upcap[i] = up_cap
    sr_downcap[i] = down_cap
    sr_up[i] = up_sr
    sr_down[i] = down_sr
    curtail[i] = curt

# Average Remuneration
print(f'Avg Remuneration S1: {np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')

#%% Strategy 2
def calculate_S2 (optimal_quantiles,p_obs,dam_price, ida_price,exc_price,def_price, afrr_price, afrr_nup, afrr_ndown,afrr_upused, afrr_downused, trup_price, trdown_price,excess,deficit,\
                         ida_forecast, idc_forecast):

    ida_s, ida_s1, ida_b, ida_b1 = 0,0,0,0
    idc_s, idc_s1, idc_b, idc_b1 = 0,0,0,0
    exc_s, exc_s1, def_b, def_b1 = 0,0,0,0
    down_cap, up_cap, down_sr, up_sr, curt = 0,0,0,0,0
    prog = 0
  
    dam_rem = optimal_quantiles * dam_price
    
    dev = ida_forecast - optimal_quantiles
    dev2 = idc_forecast - optimal_quantiles

    if dev < 0: ## Negative deviation in IDA
        if ida_price < dam_price:
            ida_b = abs(dev)
            ida_rem = - ida_b * ida_price
            dev3 = idc_forecast - optimal_quantiles + ida_b
            if dev3 > 0:
                idc_s = dev3
                idc_rem = idc_s * dam_price
                final_schedule = optimal_quantiles - ida_b + idc_s
            else:
                idc_b = abs(dev3)
                idc_rem = - idc_b * dam_price
                final_schedule = optimal_quantiles - ida_b - idc_b
            
            imbalance = p_obs - final_schedule
            exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
            total_rev = dam_rem + ida_rem + idc_rem + imb_rem
           
        else: ## No action in IDA
            if dev2 > 0:
                idc_s = dev2
                idc_rem = idc_s * dam_price
                final_schedule = optimal_quantiles + idc_s
            else:
                idc_b = abs(dev2)
                idc_rem = - idc_b * dam_price
                final_schedule = optimal_quantiles - idc_b
            
            imbalance = p_obs - final_schedule
            exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
            total_rev = dam_rem + idc_rem + imb_rem
            
            
    else: # Positive deviation hours before delivery

        if afrr_nup > min_bid and optimal_quantiles > min_bid and dev > min_bid:
            
            ### Capacity Market           
            prog = optimal_quantiles
            deviation = dev
            down_cap, up_cap, down_co, cap_rem = calculate_cap (prog, afrr_ndown, afrr_nup, deviation, afrr_price)
            
            ## IDA
            if ida_price > dam_price and up_cap < dev:
                ida_s1 = max(0, 1 * (dev - up_cap))
                ida_rem = ida_s1 * ida_price
             
            else:
                ida_s1 = 0
                ida_rem = 0
                  
            dev3 = idc_forecast - optimal_quantiles - up_cap - ida_s1
            ## IDC
            if dev3 > 0:
                idc_s1 = dev3
                idc_rem = idc_s1 * dam_price
                prog = optimal_quantiles + ida_s1 + idc_s1 ##Final Position
            else:
                idc_b1 = abs(dev3)
                idc_rem = - idc_b1 * dam_price
                prog = optimal_quantiles + ida_s1 - idc_b1
            
            ### Real-time
            avcap = p_obs - prog # Available capacity after complying with dispatch
                
            ### TSO requests for activation
            up_req = min(afrr_upused, up_cap)
            down_req = min(max(0,afrr_downused - down_co), down_cap)
                
            ### Activation of energy bids
            final_deviation, up_sr, down_sr, en_rem = calculate_en_activation(
                up_req, down_req, afrr_upused, afrr_downused, avcap, prog,
                p_obs, trup_price, trdown_price, deficit, excess)
                        
            # Imbalance Settlement      
            exc_s1, def_b1, dev_rem, curt = revenue_IS_SR (final_deviation, excess, deficit)
            total_rev = dam_rem + cap_rem + ida_rem + idc_rem +  en_rem + dev_rem
           
        else: # No participation in Reserve.
            if ida_price > dam_price:
                ida_s = dev
                ida_rem =  ida_s * ida_price
                dev3 = idc_forecast - optimal_quantiles - ida_s
                if dev3 > 0:
                    idc_s = dev3
                    idc_rem = idc_s * dam_price
                    final_schedule = optimal_quantiles + ida_s + idc_s
                else:
                    idc_b = abs(dev3)
                    idc_rem = - idc_b * dam_price
                    final_schedule = optimal_quantiles + ida_s - idc_b
            
                imbalance = p_obs - final_schedule
                exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                total_rev = dam_rem + ida_rem + idc_rem + imb_rem
                
            else: ##No action in IDA
             
                if dev2 > 0:
                    idc_s = dev2
                    idc_rem = idc_s * dam_price
                    final_schedule = optimal_quantiles + idc_s
                else:
                    idc_b = abs(dev2)
                    idc_rem = - idc_b * dam_price
                    final_schedule = optimal_quantiles  - idc_b
                
                imbalance = p_obs - final_schedule
                exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                total_rev = dam_rem + idc_rem + imb_rem
        
    return total_rev, ida_s, ida_s1, ida_b, ida_b1,idc_s, idc_s1, idc_b, idc_b1, exc_s, exc_s1, def_b, def_b1, down_cap, up_cap, down_sr, up_sr, curt

# Iterate through each time step 
for i in range(len(p_obs)):
    
    # Call the function for each time step and unpack the results
    total_rev, ida_s, ida_s1, ida_b, ida_b1, idc_s, idc_s1, idc_b, idc_b1,exc_s, exc_s1, def_b, def_b1, down_cap,\
    up_cap, down_sr, up_sr, curt = calculate_S2(optimal_quantiles[i],
        p_obs[i], dam_price[i] ,ida_price[i], exc_price[i], def_price[i], afrr_price[i],
        afrr_nup[i], afrr_ndown[i],afrr_upused[i],afrr_downused[i],trup_price[i],
        trdown_price[i] ,excess[i],deficit[i],ida_forecast[i],idc_forecast[i]
        
    )
    # Store the total revenue for the each time step
    total_revenue[i] = total_rev
    ida_sell[i] = ida_s
    ida_sell1[i] = ida_s1
    idc_sell[i] = idc_s
    idc_sell1[i] = idc_s1
    exc_sell[i] = exc_s
    exc_sell1[i] = exc_s1
    ida_buy[i] = ida_b
    ida_buy1[i] = ida_b1
    idc_buy[i] = idc_b
    idc_buy1[i] = idc_b1
    def_buy[i] = def_b
    def_buy1[i] = def_b1
    sr_upcap[i] = up_cap
    sr_downcap[i] = down_cap
    sr_up[i] = up_sr
    sr_down[i] = down_sr
    curtail[i] = curt
 
# Average Remuneration
print(f'Avg Remuneration S2:{np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')

#%% Strategy 3
def calculate_S3 (optimal_quantiles,p_obs,dam_price, ida_price,exc_price,def_price, afrr_price, afrr_nup, afrr_ndown,afrr_upused, afrr_downused, trup_price, trdown_price,excess,deficit,\
                         ida_forecast, idc_forecast):

    ida_s, ida_s1, ida_b, ida_b1 = 0,0,0,0
    idc_s, idc_s1, idc_b, idc_b1 = 0,0,0,0
    exc_s, exc_s1, def_b, def_b1 = 0,0,0,0
    down_cap, up_cap, down_sr, up_sr, curt = 0,0,0,0,0
  
    dam_rem = optimal_quantiles * dam_price
    
    dev = ida_forecast - optimal_quantiles
    dev2 = idc_forecast - optimal_quantiles

    if dev < 0: ## Negative deviation in IDA
        if ida_price < dam_price:
            ida_b = abs(dev)
            ida_rem = - ida_b * ida_price
            prog = optimal_quantiles - ida_b
            dev3 = idc_forecast - optimal_quantiles + ida_b
            if dev3 > 0:
                if afrr_nup > min_bid and prog > min_bid:
                    prog = optimal_quantiles - ida_b
                    deviation = dev3
                    down_cap, up_cap, down_co, cap_rem = calculate_cap (prog, afrr_ndown, afrr_nup, deviation, afrr_price)
                    
                    # IDC
                    if up_cap < dev3:
                        idc_s1 = max(0, dev3 - up_cap)
                        idc_rem = idc_s1 * dam_price
                    else:
                        idc_s1 = 0
                        idc_rem = 0
                      
                    prog = optimal_quantiles - ida_b + idc_s1 ## Final position before activation of bids
                
                    ### Real-time
                    avcap = p_obs - prog # Available capacity after complying with dispatch
                    
                    ### TSO requests for activation
                    up_req = min(afrr_upused, up_cap)
                    down_req = min(max(0,afrr_downused - down_co), down_cap)
                    
                    ### Activation of energy bids
                    final_deviation, up_sr, down_sr, en_rem = calculate_en_activation(
                    up_req, down_req, afrr_upused, afrr_downused, avcap, prog,
                    p_obs, trup_price, trdown_price, deficit, excess)
                    
                    # Imbalance Settlement      
                    exc_s1, def_b1, dev_rem, curt = revenue_IS_SR (final_deviation, excess, deficit)
                    total_rev = dam_rem + cap_rem + ida_rem + idc_rem +  en_rem + dev_rem
               
                else: ## Sell in IDC 
                    
                    idc_s = dev3
                    idc_rem = idc_s * dam_price
                    final_schedule = optimal_quantiles - ida_b + idc_s
                    imbalance = p_obs - final_schedule
                    exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                    total_rev = dam_rem + ida_rem + idc_rem + imb_rem
            
            else: ## Buy in IDC
                idc_b = abs(dev3)
                idc_rem = - idc_b * dam_price
                final_schedule = optimal_quantiles - ida_b - idc_b
            
                imbalance = p_obs - final_schedule
                exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                total_rev = dam_rem + ida_rem +  idc_rem + imb_rem
           
        else: ## No action in IDA
            if dev2 > 0:
                if afrr_nup > min_bid and optimal_quantiles > min_bid and dev2 > min_bid:
                    
                    prog = optimal_quantiles
                    deviation = dev2
                    down_cap, up_cap, down_co, cap_rem = calculate_cap (prog, afrr_ndown, afrr_nup, deviation, afrr_price)
            
                    # IDC
                    if up_cap < dev2:
                        idc_s1 = max(0, dev2 - up_cap)
                        idc_rem = idc_s1 * dam_price
                    else:
                        idc_s1 = 0
                        idc_rem = 0
                      
                    prog = optimal_quantiles + idc_s1 ## Final position before activation of bids
        
                    ### Real-time
                    avcap = p_obs - prog # Available capacity after complying with dispatch
                    
                    ### TSO requests for activation
                    up_req = min(afrr_upused, up_cap)
                    down_req = min(max(0,afrr_downused - down_co), down_cap)
                    
                    ### Activation of energy bids
                    final_deviation, up_sr, down_sr, en_rem = calculate_en_activation(
                    up_req, down_req, afrr_upused, afrr_downused, avcap, prog,
                    p_obs, trup_price, trdown_price, deficit, excess)
                    
                   
                    # Imbalance Settlement      
                    exc_s1, def_b1, dev_rem, curt = revenue_IS_SR (final_deviation, excess, deficit)
                    total_rev = dam_rem + cap_rem + idc_rem +  en_rem + dev_rem
                    
                    
                else:
                    idc_s = dev2
                    idc_rem = idc_s * dam_price
                    final_schedule = optimal_quantiles + idc_s
                    imbalance = p_obs - final_schedule
                    exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                    total_rev = dam_rem + idc_rem + imb_rem
                    
            else:
                idc_b = abs(dev2)
                idc_rem = - idc_b * dam_price
                final_schedule = optimal_quantiles - idc_b
                imbalance = p_obs - final_schedule
                exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                total_rev = dam_rem + idc_rem + imb_rem
            
            
    else: # No action in IDA
        if dev2 > 0:
            if afrr_nup > min_bid and optimal_quantiles > min_bid and dev2 > min_bid:    
                prog = optimal_quantiles
                deviation = dev2
                down_cap, up_cap, down_co, cap_rem = calculate_cap (prog, afrr_ndown, afrr_nup, deviation, afrr_price)
                    
                if up_cap < dev2:
                    idc_s1 = max(0, dev2 - up_cap)
                    idc_rem = idc_s1 * dam_price
                else:
                    idc_s1 = 0
                    idc_rem = 0
                  
                prog = optimal_quantiles + idc_s1 ## Final position before activation of bids
            
                ### Real-time
                avcap = p_obs - prog # Available capacity after complying with dispatch
                
                ### TSO requests for activation
                up_req = min(afrr_upused, up_cap)
                down_req = min(max(0,afrr_downused - down_co), down_cap)
                
                ### Activation of energy bids
                final_deviation, up_sr, down_sr, en_rem = calculate_en_activation(
                up_req, down_req, afrr_upused, afrr_downused, avcap, prog,
                p_obs, trup_price, trdown_price, deficit, excess)
                
                  
                # Imbalance Settlement      
                exc_s1, def_b1, dev_rem, curt = revenue_IS_SR (final_deviation, excess, deficit)
                total_rev = dam_rem + cap_rem + idc_rem +  en_rem + dev_rem
                
                
            else: ## Sell in IDC
                idc_s = dev2
                idc_rem = idc_s * dam_price
                final_schedule = optimal_quantiles + idc_s
            
                imbalance = p_obs - final_schedule
                exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
                total_rev = dam_rem + idc_rem + imb_rem
            
        else: ## Buy in IDC
            idc_b = abs(dev2)
            idc_rem = - idc_b * dam_price
            final_schedule = optimal_quantiles - idc_b
            # Imbalance Settlement
            imbalance = p_obs - final_schedule
            exc_s, def_b, imb_rem = revenue_IS (imbalance, exc_price, def_price)
            total_rev = dam_rem + idc_rem + imb_rem
                  
    return total_rev, ida_s, ida_s1, ida_b, ida_b1,idc_s, idc_s1, idc_b, idc_b1, exc_s, exc_s1, def_b, def_b1, down_cap, up_cap, down_sr, up_sr, curt 

# Iterate through each time step 
for i in range(len(p_obs)):
    
    # Call the function for each time step and unpack the results
    total_rev, ida_s, ida_s1, ida_b, ida_b1, idc_s, idc_s1, idc_b, idc_b1,exc_s, exc_s1, def_b, def_b1, down_cap,\
    up_cap, down_sr, up_sr, curt = calculate_S3(optimal_quantiles[i],
        p_obs[i], dam_price[i] ,ida_price[i], exc_price[i], def_price[i], afrr_price[i],
        afrr_nup[i], afrr_ndown[i],afrr_upused[i],afrr_downused[i],trup_price[i],
        trdown_price[i] ,excess[i],deficit[i],ida_forecast[i],idc_forecast[i]
        
    )
    # Store the total revenue for the each time step
    total_revenue[i] = total_rev
    ida_sell[i] = ida_s
    ida_sell1[i] = ida_s1
    idc_sell[i] = idc_s
    idc_sell1[i] = idc_s1
    exc_sell[i] = exc_s
    exc_sell1[i] = exc_s1
    ida_buy[i] = ida_b
    ida_buy1[i] = ida_b1
    idc_buy[i] = idc_b
    idc_buy1[i] = idc_b1
    def_buy[i] = def_b
    def_buy1[i] = def_b1
    sr_upcap[i] = up_cap
    sr_downcap[i] = down_cap
    sr_up[i] = up_sr
    sr_down[i] = down_sr
    curtail[i] = curt

# Average Remuneration
print(f'Avg Remuneration S3: {np.sum(total_revenue) / np.sum(p_obs):.2f} €/MWh')

