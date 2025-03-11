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
epsilon = 1e-12 ## Small threshold to handle floating-point precision errors


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
final_balance = np.zeros_like(p_obs, dtype=float)

#%% Bid in DAM

threshold = 43.43
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
def calculate_S2 (optimal_quantiles,p_obs,dam_price, ida_price,exc_price,def_price, afrr_price, afrr_nup, afrr_ndown,afrr_upused, afrr_downused, trup_price, trdown_price,excess,deficit,\
                         ida_forecast, idc_forecast):

    ida_s, ida_s1, ida_b, ida_b1 = 0,0,0,0
    idc_s, idc_s1, idc_b, idc_b1 = 0,0,0,0
    exc_s, exc_s1, def_b, def_b1 = 0,0,0,0
    down_cap, up_cap, down_sr, up_sr, curt = 0,0,0,0,0
    prog = 0

    dam_rem = optimal_quantiles * dam_price
    
    dev = ida_forecast - optimal_quantiles ## Deviation hours before delivery
    dev2 = idc_forecast - optimal_quantiles ## Deviation 15min before delivery

    if dev < 0: ## Negative deviation in IDA
        
        if ida_price < dam_price:
            ida_b = 1 * abs(dev)
            ida_rem = - ida_b * ida_price
            dev3 = idc_forecast - optimal_quantiles + ida_b
            if dev3 > 0:
                idc_s = dev3
                idc_rem = idc_s * dam_price
                imbalance = p_obs - optimal_quantiles + ida_b - idc_s
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem + ida_rem + idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem + ida_rem + idc_rem + dev_rem
            else:
                idc_b = abs(dev3)
                idc_rem = - idc_b * dam_price
                imbalance = p_obs - optimal_quantiles + ida_b + idc_b
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem + ida_rem + idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem + ida_rem + idc_rem + dev_rem

        else: ## No action in IDA
            if dev2 > 0:
                idc_s = dev2
                idc_rem = idc_s * dam_price
                imbalance = p_obs - optimal_quantiles - idc_s
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem + idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem + idc_rem + dev_rem
            else:
                idc_b = abs(dev2)
                idc_rem = - idc_b * dam_price
                imbalance = p_obs - optimal_quantiles + idc_b
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem +  idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem + idc_rem + dev_rem
        
    
    else: # Positive deviation hours before delivery

        if afrr_nup > 0 and optimal_quantiles > min_bid and dev > min_bid:
            
            ### Capacity Market [MW]
             
            ## Can't offer more downward capacity than the current position
            down_cap = min(dev/2, afrr_ndown, optimal_quantiles)
            up_cap = min(dev, afrr_nup, down_cap*2)
            cap_rem = (up_cap * afrr_price) + (down_cap * afrr_price)
            down_co = afrr_ndown - down_cap  
                
          
            if ida_price > dam_price and up_cap < dev:
                ida_s1 = max(0, 1 * (dev - up_cap))
                ida_rem = ida_s1 * ida_price
             
            else:
                ida_s1 = 0
                ida_rem = 0
                  
            dev3 = idc_forecast - optimal_quantiles - up_cap - ida_s1
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
            
            if avcap >= up_req: ## Available capacity is higher than requested for UP activ
                if up_req > (down_req*2) or down_req == 0: ## Activation only in Up direction 
                    up_sr = min(afrr_upused, up_cap)
                    en_rem = up_sr * trup_price
                    imbalance = p_obs - prog - up_sr
                    if imbalance > 0:
                        if excess > 0:
                            exc_s1 = imbalance
                            dev_rem = exc_s1 * excess
                            total_rev = dam_rem + cap_rem + en_rem + dev_rem+ ida_rem + idc_rem
                        else:
                            curt = imbalance
                            total_rev = dam_rem + cap_rem  + en_rem+ ida_rem + idc_rem
                    else:
                        def_b1 = abs(imbalance)
                        def_rem = - def_b1 * deficit
                        total_rev = dam_rem + cap_rem + en_rem + def_rem+ ida_rem + idc_rem
                
                else: 
                    
                    up_sr = up_req
                    if prog + (2*up_sr) <= p_obs: ## Activation in both directions. 30min UP and 30min DOWN
                        down_sr = down_req
                    else:
                        down_sr = 0
                    
                    en_rem = (up_sr * trup_price) - (down_sr * trdown_price)
                    imbalance = p_obs - prog - up_sr  + down_sr
                       
                    if imbalance > 0:
                        if excess > 0:
                            exc_s1 = imbalance
                            dev_rem = exc_s1 * excess
                            total_rev = dam_rem + cap_rem  + en_rem + dev_rem + ida_rem + idc_rem
                        else:
                            curt = imbalance
                            total_rev = dam_rem + cap_rem + en_rem + ida_rem + idc_rem
                    else:
                        def_b1 = abs(imbalance)
                        def_rem = - def_b1 * deficit
                        total_rev = dam_rem + cap_rem + en_rem + def_rem + ida_rem + idc_rem
                    
                
                    
            else: ## Available capacity not enough to activate the requested in UP direction
                 
                if avcap > 0:
                    up_sr = avcap
                    en_rem = up_sr * trup_price
                    def_b1 = abs(up_req - up_sr)
                    def_rem = - def_b1 * deficit
                    total_rev = dam_rem + ida_rem +  cap_rem + en_rem + def_rem + idc_rem
         
                else:
                    def_b1 = abs(p_obs - prog - up_req)
                    def_rem = - def_b1 * deficit
                    total_rev = dam_rem + ida_rem + cap_rem + def_rem + idc_rem
                        
                                
                     
        else: # No participation in Reserve. 
            if ida_price > dam_price:
                ida_s = dev
                ida_rem = ida_s * ida_price
                dev3 = idc_forecast - optimal_quantiles - ida_s
                if dev3 > 0:
                    idc_s = dev3
                    idc_rem = idc_s * dam_price
                    imbalance = p_obs - optimal_quantiles - ida_s - idc_s
                    if imbalance > 0:
                        exc_s = imbalance
                        dev_rem = exc_s * exc_price
                        total_rev = dam_rem + ida_rem + idc_rem +  dev_rem
                    else:
                        def_b = abs(imbalance)
                        dev_rem = - def_b * def_price
                        total_rev = dam_rem + ida_rem + idc_rem + dev_rem
                else:
                    idc_b = abs(dev3)
                    idc_rem = - idc_b * dam_price
                    imbalance = p_obs - optimal_quantiles - ida_s + idc_b
                    if imbalance > 0:
                        exc_s = imbalance
                        dev_rem = exc_s * exc_price
                        total_rev = dam_rem + ida_rem + idc_rem +  dev_rem
                    else:
                        def_b = abs(imbalance)
                        dev_rem = - def_b * def_price
                        total_rev = dam_rem + ida_rem + idc_rem + dev_rem
                
        
                
            else: ### No action in IDA
                if dev2 > 0:
                    idc_s = dev2
                    idc_rem = idc_s * dam_price
                    imbalance = p_obs - optimal_quantiles - idc_s
                    if imbalance > 0:
                        exc_s = imbalance
                        dev_rem = exc_s * exc_price
                        total_rev = dam_rem + idc_rem +  dev_rem
                    else:
                        def_b = abs(imbalance)
                        dev_rem = - def_b * def_price
                        total_rev = dam_rem + idc_rem + dev_rem
                else:
                    idc_b = abs(dev2)
                    idc_rem = - idc_b * dam_price
                    imbalance = p_obs - optimal_quantiles + idc_b
                    if imbalance > 0:
                        exc_s = imbalance
                        dev_rem = exc_s * exc_price
                        total_rev = dam_rem +  idc_rem +  dev_rem
                    else:
                        def_b = abs(imbalance)
                        dev_rem = - def_b * def_price
                        total_rev = dam_rem + idc_rem + dev_rem
          
               
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

#%%
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
            ida_b = 1 * abs(dev)
            ida_rem = - ida_b * ida_price
            dev3 = idc_forecast - optimal_quantiles + ida_b
            if dev3 > 0:
                ### Participation in reserve
                if afrr_nup > 0 and dev3 > min_bid:
                    prog = optimal_quantiles - ida_b
                    down_cap = min(prog, afrr_ndown, dev3/2)
                    up_cap = min(afrr_nup, down_cap*2)
                    cap_rem = (up_cap * afrr_price) + (down_cap * afrr_price)
                    down_co = afrr_ndown - down_cap
                    
                    if up_cap < dev3:
                        idc_s1 = max(0, dev3 - up_cap)
                        idc_rem = idc_s1 * dam_price
               
                    else:
                        idc_s1 = 0
                        idc_rem = 0

                    prog = optimal_quantiles - ida_b + idc_s1
                    avcap = p_obs - prog
                    
                    ### TSO requests
                    up_req = min(afrr_upused, up_cap)
                    down_req = min(max(0, afrr_downused-down_co),down_cap)
                    
                    ###Activation
                    if avcap >= up_req:
                        if up_req > (down_req*2)  or down_req == 0:
                            up_sr = min(afrr_upused, up_cap)
                            en_rem = up_sr * trup_price
                            imbalance = p_obs - prog - up_sr
                            if imbalance > 0:
                                if excess > 0:
                                    exc_s1 = imbalance
                                    dev_rem = exc_s1 * excess
                                    total_rev = dam_rem + cap_rem + en_rem + dev_rem + ida_rem + idc_rem
                                else:
                                    curt = imbalance
                                    total_rev = dam_rem + cap_rem  + en_rem+ ida_rem + idc_rem
                            else:
                                def_b1 = abs(imbalance)
                                def_rem = - def_b1 * deficit
                                total_rev = dam_rem + cap_rem + en_rem + def_rem+ ida_rem + idc_rem
                            
                        else:
                            up_sr = up_req
                            if prog + (2*up_sr) <= p_obs:
                                down_sr = down_req
                            else:
                                down_sr = 0
                            
                            en_rem = (up_sr * trup_price) - (down_sr * trdown_price)
                            imbalance = p_obs - prog - up_sr  + down_sr
                               
                            if imbalance > 0:
                                if excess > 0:
                                    exc_s1 = imbalance
                                    dev_rem = exc_s1 * excess
                                    total_rev = dam_rem + cap_rem  + en_rem + dev_rem + ida_rem + idc_rem
                                else:
                                    curt = imbalance
                                    total_rev = dam_rem + cap_rem + en_rem + ida_rem + idc_rem
                            else:
                                def_b1 = abs(imbalance)
                                def_rem = - def_b1 * deficit
                                total_rev = dam_rem + cap_rem + en_rem + def_rem + ida_rem + idc_rem
                                    
                        
                    else:
                        if avcap > 0:
                            up_sr = avcap
                            en_rem = up_sr * trup_price
                            def_b1 = abs(up_req - up_sr)
                            def_rem = - def_b1 * deficit
                            total_rev = dam_rem + ida_rem +  cap_rem + en_rem + def_rem + idc_rem
                 
                        else:
                            def_b1 = abs(p_obs - prog - up_req)
                            def_rem = - def_b1 * deficit
                            total_rev = dam_rem + ida_rem + cap_rem + def_rem + idc_rem
                    
                    
                else: ## No reserve in this hour
                    idc_s = dev3
                    idc_rem = idc_s * dam_price
                    imbalance = p_obs - optimal_quantiles + ida_b - idc_s
                    if imbalance > 0:
                        exc_s = imbalance
                        dev_rem = exc_s * exc_price
                        total_rev = dam_rem + ida_rem + idc_rem +  dev_rem
                    else:
                        def_b = abs(imbalance)
                        dev_rem = - def_b * def_price
                        total_rev = dam_rem + ida_rem + idc_rem + dev_rem
            
            else:
                idc_b = abs(dev3)
                idc_rem = - idc_b * dam_price
                imbalance = p_obs - optimal_quantiles + ida_b + idc_b
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem + ida_rem + idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem + ida_rem + idc_rem + dev_rem

        else: ## No action in IDA
            if dev2 > 0:   
                
                ## PARTICIPATION IN SR 
                if afrr_nup > 0 and dev2 > min_bid:
                    prog = optimal_quantiles - ida_b
                    down_cap = min(prog, afrr_ndown, dev2/2)
                    up_cap = min(afrr_nup, down_cap*2)
                    cap_rem = (up_cap * afrr_price) + (down_cap * afrr_price)
                    down_co = afrr_ndown - down_cap
                    
                    if up_cap < dev2:
                        idc_s1 = max(0, dev2 - up_cap)
                        idc_rem = idc_s1 * dam_price
               
                    else:
                        idc_s1 = 0
                        idc_rem = 0

                    prog = optimal_quantiles  + idc_s1
                    avcap = p_obs - prog
                    
                    ### TSO requests
                    up_req = min(afrr_upused, up_cap)
                    down_req = min(max(0, afrr_downused-down_co),down_cap)
                    
                    ###Activation
                    if avcap >= up_req:
                        if up_req > (down_req*2)  or down_req == 0:
                            up_sr = min(afrr_upused, up_cap)
                            en_rem = up_sr * trup_price
                            imbalance = p_obs - prog - up_sr
                            if imbalance > 0:
                                if excess > 0:
                                    exc_s1 = imbalance
                                    dev_rem = exc_s1 * excess
                                    total_rev = dam_rem + cap_rem + en_rem + dev_rem  + idc_rem
                                else:
                                    curt = imbalance
                                    total_rev = dam_rem + cap_rem  + en_rem +  idc_rem
                            else:
                                def_b1 = abs(imbalance)
                                def_rem = - def_b1 * deficit
                                total_rev = dam_rem + cap_rem + en_rem + def_rem + idc_rem
                            
                        else:
                            up_sr = up_req
                            if prog + (2*up_sr) <= p_obs:
                                down_sr = down_req
                            else:
                                down_sr = 0
                            
                            en_rem = (up_sr * trup_price) - (down_sr * trdown_price)
                            imbalance = p_obs - prog - up_sr  + down_sr
                               
                            if imbalance > 0:
                                if excess > 0:
                                    exc_s1 = imbalance
                                    dev_rem = exc_s1 * excess
                                    total_rev = dam_rem + cap_rem  + en_rem + dev_rem + idc_rem
                                else:
                                    curt = imbalance
                                    total_rev = dam_rem + cap_rem + en_rem + idc_rem
                            else:
                                def_b1 = abs(imbalance)
                                def_rem = - def_b1 * deficit
                                total_rev = dam_rem + cap_rem + en_rem + def_rem + idc_rem
                                    
                        
                    else:
                        if avcap > 0:
                            up_sr = avcap
                            en_rem = up_sr * trup_price
                            def_b1 = abs(up_req - up_sr)
                            def_rem = - def_b1 * deficit
                            total_rev = dam_rem +  cap_rem + en_rem + def_rem + idc_rem
                 
                        else:
                            def_b1 = abs(p_obs - prog - up_req)
                            def_rem = - def_b1 * deficit
                            total_rev = dam_rem +  cap_rem + def_rem + idc_rem
                
           
                    
                else: ## No participation in reserve 
                    
                    idc_s = dev2
                    idc_rem = idc_s * dam_price
                    imbalance = p_obs - optimal_quantiles  - idc_s
                    if imbalance > 0:
                        exc_s = imbalance
                        dev_rem = exc_s * exc_price
                        total_rev = dam_rem + idc_rem +  dev_rem
                    else:
                        def_b = abs(imbalance)
                        dev_rem = - def_b * def_price
                        total_rev = dam_rem  + idc_rem + dev_rem
                
            else:
                idc_b = abs(dev2)
                idc_rem = - idc_b * dam_price
                imbalance = p_obs - optimal_quantiles + idc_b
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem +  idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem + idc_rem + dev_rem
        
    
    else: # Positive deviation in IDA. No action
        if dev2 > 0: 
            ### Participation in RESERVE
            
            if afrr_nup > 0 and dev2 > min_bid:
              
                    prog = optimal_quantiles - ida_b
                    down_cap = min(prog, afrr_ndown, dev2/2)
                    up_cap = min(afrr_nup, down_cap*2)
                    cap_rem = (up_cap * afrr_price) + (down_cap * afrr_price)
                    down_co = afrr_ndown - down_cap
                    
                    if up_cap < dev2:
                        idc_s1 = max(0, dev2 - up_cap)
                        idc_rem = idc_s1 * dam_price
               
                    else:
                        idc_s1 = 0
                        idc_rem = 0

                    prog = optimal_quantiles  + idc_s1
                    avcap = p_obs - prog
                    
                    ### TSO requests
                    up_req = min(afrr_upused, up_cap)
                    down_req = min(max(0, afrr_downused-down_co),down_cap)
                    
                    ###Activation
                    if avcap >= up_req:
                        if up_req > (down_req*2)  or down_req == 0:
                            up_sr = min(afrr_upused, up_cap)
                            en_rem = up_sr * trup_price
                            imbalance = p_obs - prog - up_sr
                            if imbalance > 0:
                                if excess > 0:
                                    exc_s1 = imbalance
                                    dev_rem = exc_s1 * excess
                                    total_rev = dam_rem + cap_rem + en_rem + dev_rem  + idc_rem
                                else:
                                    curt = imbalance
                                    total_rev = dam_rem + cap_rem  + en_rem +  idc_rem
                            else:
                                def_b1 = abs(imbalance)
                                def_rem = - def_b1 * deficit
                                total_rev = dam_rem + cap_rem + en_rem + def_rem + idc_rem
                            
                        else:
                            up_sr = up_req
                            if prog + (2*up_sr) <= p_obs:
                                down_sr = down_req
                            else:
                                down_sr = 0
                            
                            en_rem = (up_sr * trup_price) - (down_sr * trdown_price)
                            imbalance = p_obs - prog - up_sr  + down_sr
                               
                            if imbalance > 0:
                                if excess > 0:
                                    exc_s1 = imbalance
                                    dev_rem = exc_s1 * excess
                                    total_rev = dam_rem + cap_rem  + en_rem + dev_rem + idc_rem
                                else:
                                    curt = imbalance
                                    total_rev = dam_rem + cap_rem + en_rem + idc_rem
                            else:
                                def_b1 = abs(imbalance)
                                def_rem = - def_b1 * deficit
                                total_rev = dam_rem + cap_rem + en_rem + def_rem + idc_rem
                                    
                        
                    else:
                        if avcap > 0:
                            up_sr = avcap
                            en_rem = up_sr * trup_price
                            def_b1 = abs(up_req - up_sr)
                            def_rem = - def_b1 * deficit
                            total_rev = dam_rem +  cap_rem + en_rem + def_rem + idc_rem
                 
                        else:
                            def_b1 = abs(p_obs - prog - up_req)
                            def_rem = - def_b1 * deficit
                            total_rev = dam_rem +  cap_rem + def_rem + idc_rem
                
                
                
            else: ## No participation in reserve 
                idc_s = dev2
                idc_rem = idc_s * dam_price
                imbalance = p_obs - optimal_quantiles  - idc_s
                if imbalance > 0:
                    exc_s = imbalance
                    dev_rem = exc_s * exc_price
                    total_rev = dam_rem + idc_rem +  dev_rem
                else:
                    def_b = abs(imbalance)
                    dev_rem = - def_b * def_price
                    total_rev = dam_rem  + idc_rem + dev_rem
 
 
        else:
            idc_b = abs(dev2)
            idc_rem = - idc_b * dam_price
            imbalance = p_obs - optimal_quantiles + idc_b
            if imbalance > 0:
                exc_s = imbalance
                dev_rem = exc_s * exc_price
                total_rev = dam_rem +  idc_rem +  dev_rem
            else:
                def_b = abs(imbalance)
                dev_rem = - def_b * def_price
                total_rev = dam_rem + idc_rem + dev_rem
          
               
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