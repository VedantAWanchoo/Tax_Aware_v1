import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
from scipy.stats import skew, kurtosis



st.set_page_config(layout='wide')
st.title('Tax-Efficient Portfolio Transition (Long Only vs. Relaxed Constraint)')



# User Inputs
buy_and_hold_months = st.sidebar.number_input('Buy and Hold Months', min_value=12, max_value=144, value=60, step=12)
lookback_months = st.sidebar.number_input('Lookback Months', value=11, step=1)
skip_months = st.sidebar.number_input('Skip Months', value=1, step=1)
forward_months = st.sidebar.number_input('Forward Months', value=1, step=1)
Portfolio_N = st.sidebar.number_input('Number of Stocks in Portfolio', value=50, step=1)
trade_cost_rate = st.sidebar.number_input('Transaction Cost per trade (%)', value=0.3, step=0.1, format='%.2f') / 100
LTCG = st.sidebar.number_input('Long Term Capital Gains Rate (%)', value=15.0, step=1.0, format='%.1f') / 100
STCG = st.sidebar.number_input('Short Term Capital Gains Rate (%)', value=30.0, step=1.0, format='%.1f') / 100
Relaxed_Contraint_Bin = st.sidebar.slider('Relaxed Constraint Bin', min_value=1, max_value=10, value=7)
Factor_Score_Bins = 10
total_lookback = lookback_months + skip_months



if st.sidebar.button('Run'):


    #data=pd.read_excel(r'C:\Users\Vedant Wanchoo\Desktop\CGS 2020\AQR Inquiry\Tax Aware Strategies\S&P_500_2025-06-13_Cleaned.xlsx')

    #data=pd.read_excel(r'C:\Users\Vedant Wanchoo\Desktop\CGS 2020\Streamlit_Explore\Tax Aware Trial\S&P_500_2025-06-13_Cleaned.xlsx')

    data = pd.read_excel('S&P_500_2025-06-13_Cleaned.xlsx')


    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data1 = data.pct_change()
    data1 = data1.iloc[1:]  # This removes only the first row
    data1.index = pd.date_range(start=(data1.index[0] - pd.DateOffset(months=1)), periods=len(data1), freq='MS')



    #Long_Only_Tax_Agnostic_Buy_And_Hold_+_Momentum


    #Overall Assumptions

    #buy_and_hold_months = 12*5
    #lookback_months = 11
    #skip_months = 1
    #forward_months = 1
    #total_lookback = lookback_months + skip_months  # 12 months
    #Portfolio_N = 50  # Number of stocks in portfolio 50
    #trade_cost_rate = 0.003  # 0.3% per trade
    #Factor_Score_Bins = 10
    #LTCG = 0.15    #0.15  
    #STCG = 0.30    #0.30  
    #Relaxed_Contraint_Bin = 7

    #Buy and Hold

    # Step 1: Extract 12 months of returns (Feb 2010 to Jan 2011)
    first_12_returns = data1.iloc[0:buy_and_hold_months]  # Feb 2010 to Jan 2011

    # Step 2: Filter to only stocks with no NaNs over these 12 months
    clean_stocks = first_12_returns.dropna(axis=1)
    eligible_stocks = clean_stocks.columns[:Portfolio_N]


    inPortfolio = pd.DataFrame(0, index=data1.index, columns=data1.columns)
    inPortfolio.loc[data1.index[:buy_and_hold_months], eligible_stocks] = 1

    # Step 3: Restrict returns to those 30 clean stocks
    first_12_returns = first_12_returns[eligible_stocks]

    # Step 4: Add dummy Jan 2010 row with 0% return
    jan_2010_date = (data1.index.min() - pd.DateOffset(months=1)).replace(day=1)  #pd.to_datetime('2010-01-01')
    jan_zero_row = pd.DataFrame([[0] * Portfolio_N], columns=eligible_stocks, index=[jan_2010_date])

    # Step 5: Combine rows to get Jan + 12 months
    returns_with_jan = pd.concat([jan_zero_row, first_12_returns])

    # Step 6: Compound returns to get NAV per stock
    cumulative_nav = (1 + returns_with_jan).cumprod()

    # Step 7: Portfolio NAV = equal-weighted average
    portfolio_nav = cumulative_nav.mean(axis=1)

    # Step 8: Monthly portfolio returns from NAV (Feb 2010 to Jan 2011)
    portfolio_returns = portfolio_nav.pct_change().iloc[1:]
    portfolio_returns.index = data1.index[0:buy_and_hold_months]  # Feb 2010 to Jan 2011

    # Step 9: Apply transaction cost only in Feb 2010 (first month)
    transaction_cost_series = [trade_cost_rate] + [0] * (len(portfolio_returns) - 1)
    post_tc_returns = portfolio_returns.copy()
    post_tc_returns.iloc[0] -= trade_cost_rate

    # Step 10: Turnover = 1.0 in Feb 2010, 0 afterward
    turnover_series = [1.0] + [0] * (len(portfolio_returns) - 1)

    # Step 11: Final DataFrame
    buy_hold_df = pd.DataFrame({
        'Momentum_Return': portfolio_returns.values,
        'Transaction_Cost': transaction_cost_series,
        'Post_T.Cost_Return': post_tc_returns.values,
        'Turnover': turnover_series,
        'Portfolio_N': [Portfolio_N] * len(portfolio_returns),
        'Filtered_Universe_N': [clean_stocks.shape[1]] * len(portfolio_returns),
        'Portfolio_Mom_Score': [np.nan] * len(portfolio_returns),
        'Universe_Mom_Score': [np.nan] * len(portfolio_returns)}, index=portfolio_returns.index)

    buy_hold_df.index.name = "Date"



    #Monthly momentum

    momentum_returns = []
    momentum_dates = []
    valid_stock_counts = []
    portfolio_stock_counts = []
    portfolio_momentum_scores = []
    universe_momentum_scores = []

    turnovers = []
    transaction_costs = []
    net_returns = []


    prev_portfolio = set(eligible_stocks)  # For tracking changes from buy and hold
    prev_nav = cumulative_nav.iloc[-1].copy()


    for i in range(buy_and_hold_months, len(data1) - forward_months + 1):

        start_idx = i - total_lookback  # t−12
        end_idx = i                     # t (month of forward return)

        # Sub-periods
        lookback_data = data1.iloc[start_idx : start_idx + lookback_months]  # t−12 to t−2
        forward_return = (1 + data1.iloc[i : i + forward_months]).prod() - 1 #data1.iloc[end_idx]  # return at month t

        # Drop stocks with any NaNs in the full 13-month window
        valid_stocks = data1.iloc[start_idx:end_idx + forward_months].dropna(axis=1).columns

        # Momentum signal: cumulative return over lookback
        cumulative_returns = (1 + lookback_data[valid_stocks]).prod() - 1

        # Top N stocks
        top_stocks = cumulative_returns.sort_values(ascending=False).head(Portfolio_N).index

        inPortfolio.loc[data1.index[end_idx], top_stocks] = 1

        # Average forward return of top N stocks
        portfolio_return = forward_return[top_stocks].mean()

        # Average Portfolio and Universe momentum score
        bins = pd.qcut(cumulative_returns, Factor_Score_Bins, labels=False) + 1
        portfolio_score = bins[top_stocks].mean()
        universe_score = bins.mean()

        #Turnover, transaction costs
        current_portfolio = set(top_stocks)
        new_stocks = current_portfolio - prev_portfolio
        num_new_stocks = len(new_stocks)
        new_stock_turnover = num_new_stocks / Portfolio_N

        overlapping_stocks = current_portfolio & prev_portfolio
        full_prev_weights = prev_nav / prev_nav.sum() 

        if overlapping_stocks:

            # Extract only overlapping stock weights
            prev_weights = full_prev_weights[list(overlapping_stocks)]

            # Target new equal weights for overlapping stocks
            new_weights = pd.Series(1 / Portfolio_N, index=overlapping_stocks)

            # Resizing turnover
            resize_turnover = ((prev_weights - new_weights).abs().sum())/2
        else:
            resize_turnover = 0

        turnover = new_stock_turnover + resize_turnover
        cost = 2 * turnover * trade_cost_rate
        net_return = portfolio_return - cost

        prev_nav = (1 + data1[top_stocks].iloc[i : i + forward_months]).prod() # Lagged return for next iteration
        prev_portfolio = current_portfolio  # Lagged portfolio for next iteration


        # Append results
        valid_stock_counts.append(len(valid_stocks))  # total valid stocks
        portfolio_stock_counts.append(len(top_stocks))  # size of portfolio (could be < 30)
        momentum_returns.append(portfolio_return)
        momentum_dates.append(data1.index[end_idx])  # label by forward return month
        portfolio_momentum_scores.append(portfolio_score)
        universe_momentum_scores.append(universe_score)
        turnovers.append(turnover)
        transaction_costs.append(cost)
        net_returns.append(net_return)


    # Construct result DataFrame with additional info

    momentum_df = pd.DataFrame({'Date': momentum_dates,'Momentum_Return': momentum_returns,'Transaction_Cost': transaction_costs,'Post_T.Cost_Return': net_returns,'Turnover': turnovers,'Portfolio_N': portfolio_stock_counts,'Filtered_Universe_N': valid_stock_counts,'Portfolio_Mom_Score': portfolio_momentum_scores,'Universe_Mom_Score': universe_momentum_scores})

    momentum_df.set_index('Date', inplace=True)


    full_df = pd.concat([buy_hold_df, momentum_df])
    full_df.sort_index(inplace=True)



    #Long_Only_Tax_Agnostic_Taxation_Impact

    tax_portfolio = inPortfolio.copy()
    tax_portfolio = tax_portfolio.replace(0, np.nan)
    tax_portfolio = tax_portfolio.where(pd.isna(tax_portfolio), data1[tax_portfolio.columns])




    trade_cycles = []

    # Iterate over each stock
    for stock in tax_portfolio.columns:
        returns = tax_portfolio[stock]
        in_trade = False
        start_date = None
        cum_return = 1.0
        months_held = 0

        for date, value in returns.items():
            if pd.notna(value):
                if not in_trade:
                    in_trade = True
                    start_date = date
                    cum_return = 1 + value
                    months_held = 1
                else:
                    cum_return *= (1 + value)
                    months_held += 1
            else:
                if in_trade:
                    trade_cycles.append({'Stock': stock, 'Open_Date': start_date,'Close_Date': prev_date,'Total_Return': cum_return - 1,'Time_Period_Months': months_held})
                    in_trade = False
                    cum_return = 1.0
                    months_held = 0
            prev_date = date


    tax_portfolio1 = pd.DataFrame(trade_cycles)



    tax_portfolio1["Year_End"] = tax_portfolio1["Close_Date"].apply(lambda x: pd.Timestamp(year=x.year, month=12, day=1))

    tax_portfolio1["LongTerm"] = (tax_portfolio1["Total_Return"] / Portfolio_N) * (tax_portfolio1["Time_Period_Months"] >= 12)

    tax_portfolio1["ShortTerm"] = (tax_portfolio1["Total_Return"] / Portfolio_N) * (tax_portfolio1["Time_Period_Months"] < 12)

    tax_portfolio2 = tax_portfolio1.groupby("Year_End")[["LongTerm", "ShortTerm"]].sum().reset_index()


    conditions = [
        (tax_portfolio2["LongTerm"] >= 0) & (tax_portfolio2["ShortTerm"] >= 0),  # both gains
        (tax_portfolio2["LongTerm"] >= 0) & (tax_portfolio2["ShortTerm"] < 0),   # LT gain, ST loss
        (tax_portfolio2["ShortTerm"] >= 0) & (tax_portfolio2["LongTerm"] < 0),   # ST gain, LT loss
        (tax_portfolio2["LongTerm"] < 0) & (tax_portfolio2["ShortTerm"] < 0)]     # both losses

    choices = [
        tax_portfolio2["LongTerm"] * LTCG + tax_portfolio2["ShortTerm"] * STCG,
        (tax_portfolio2["LongTerm"] + tax_portfolio2["ShortTerm"]) * LTCG,
        (tax_portfolio2["ShortTerm"] + tax_portfolio2["LongTerm"]) * STCG,
        tax_portfolio2["LongTerm"] * LTCG + tax_portfolio2["ShortTerm"] * STCG]

    # Assign calculated tax liability without forcing minimum 0
    tax_portfolio2["tax_liability"] = pd.Series(np.select(conditions, choices, default=0))


    # Step 1: Create a DataFrame from tax_portfolio2 with Year_End as index
    tax_liability_df = tax_portfolio2[["Year_End", "LongTerm","ShortTerm","tax_liability"]].set_index("Year_End")

    # Step 2: Map tax liability to full_df based on index (Date)
    full_df["LongTerm"] = full_df.index.map(tax_liability_df["LongTerm"])
    full_df["ShortTerm"] = full_df.index.map(tax_liability_df["ShortTerm"])
    full_df["Tax_Loss"] = full_df.index.map(tax_liability_df["tax_liability"])

    full_df[["LongTerm", "ShortTerm","Tax_Loss"]] = full_df[["LongTerm", "ShortTerm","Tax_Loss"]].fillna(0)

    full_df["Post_Tax_Return"] = full_df["Post_T.Cost_Return"] - full_df["Tax_Loss"]



    # Column Orders

    cols = list(full_df.columns)
    last_two = cols[-2:]

    for col in last_two:
        cols.remove(col)

    cols[3:3] = last_two
    full_df = full_df[cols]


    long_only_df = full_df.copy()
    long_only_inPortfolio = inPortfolio.copy() 



    #Relaxed_Constraint_Tax_Agnostic_Buy_And_Hold_+_Momentum


    #Buy and Hold

    # Step 1: Extract 12 months of returns (Feb 2010 to Jan 2011)
    first_12_returns = data1.iloc[0:buy_and_hold_months]  # Feb 2010 to Jan 2011

    # Step 2: Filter to only stocks with no NaNs over these 12 months
    clean_stocks = first_12_returns.dropna(axis=1)
    eligible_stocks = clean_stocks.columns[:Portfolio_N]


    inPortfolio = pd.DataFrame(0, index=data1.index, columns=data1.columns)
    inPortfolio.loc[data1.index[:buy_and_hold_months], eligible_stocks] = 1

    # Step 3: Restrict returns to those 30 clean stocks
    first_12_returns = first_12_returns[eligible_stocks]

    # Step 4: Add dummy Jan 2010 row with 0% return
    jan_2010_date = (data1.index.min() - pd.DateOffset(months=1)).replace(day=1)  #pd.to_datetime('2010-01-01')
    jan_zero_row = pd.DataFrame([[0] * Portfolio_N], columns=eligible_stocks, index=[jan_2010_date])

    # Step 5: Combine rows to get Jan + 12 months
    returns_with_jan = pd.concat([jan_zero_row, first_12_returns])

    # Step 6: Compound returns to get NAV per stock
    cumulative_nav = (1 + returns_with_jan).cumprod()

    # Step 7: Portfolio NAV = equal-weighted average
    portfolio_nav = cumulative_nav.mean(axis=1)

    # Step 8: Monthly portfolio returns from NAV (Feb 2010 to Jan 2011)
    portfolio_returns = portfolio_nav.pct_change().iloc[1:]
    portfolio_returns.index = data1.index[0:buy_and_hold_months]  # Feb 2010 to Jan 2011

    # Step 9: Apply transaction cost only in Feb 2010 (first month)
    transaction_cost_series = [trade_cost_rate] + [0] * (len(portfolio_returns) - 1)
    post_tc_returns = portfolio_returns.copy()
    post_tc_returns.iloc[0] -= trade_cost_rate

    # Step 10: Turnover = 1.0 in Feb 2010, 0 afterward
    turnover_series = [1.0] + [0] * (len(portfolio_returns) - 1)

    # Step 11: Final DataFrame
    buy_hold_df = pd.DataFrame({
        'Momentum_Return': portfolio_returns.values,
        'Transaction_Cost': transaction_cost_series,
        'Post_T.Cost_Return': post_tc_returns.values,
        'Turnover': turnover_series,
        'Portfolio_N': [Portfolio_N] * len(portfolio_returns),
        'Filtered_Universe_N': [clean_stocks.shape[1]] * len(portfolio_returns),
        'Portfolio_Mom_Score': [np.nan] * len(portfolio_returns),
        'Universe_Mom_Score': [np.nan] * len(portfolio_returns)
    }, index=portfolio_returns.index)

    buy_hold_df.index.name = "Date"



    #Monthly momentum

    momentum_returns = []
    momentum_dates = []
    valid_stock_counts = []
    portfolio_stock_counts = []
    portfolio_momentum_scores = []
    universe_momentum_scores = []

    turnovers = []
    transaction_costs = []
    net_returns = []


    #Portfolio_N = int(Portfolio_N*Leverage) 

    prev_portfolio = set(eligible_stocks)  # For tracking changes from buy and hold
    prev_nav = cumulative_nav.iloc[-1].copy()


    for i in range(buy_and_hold_months, len(data1) - forward_months + 1):

        start_idx = i - total_lookback  # t−12
        end_idx = i                     # t (month of forward return)

        # Sub-periods
        lookback_data = data1.iloc[start_idx : start_idx + lookback_months]  # t−12 to t−2
        forward_return = (1 + data1.iloc[i : i + forward_months]).prod() - 1 #data1.iloc[end_idx]  # return at month t

        # Drop stocks with any NaNs in the full 13-month window
        valid_stocks = data1.iloc[start_idx:end_idx + forward_months].dropna(axis=1).columns

        # Momentum signal: cumulative return over lookback
        cumulative_returns = (1 + lookback_data[valid_stocks]).prod() - 1


        # Relaxed Constraint Picking

        bins = pd.qcut(cumulative_returns, Factor_Score_Bins, labels=False) + 1

        retained_stocks = [stock for stock in prev_portfolio if stock in bins and bins[stock] >= Relaxed_Contraint_Bin] 

        stocks_sorted = cumulative_returns.sort_values(ascending=False).index

        new_pick_universe = [stock for stock in stocks_sorted if stock not in retained_stocks]

        n_new = Portfolio_N - len(retained_stocks)

        new_picks_final = new_pick_universe[:n_new]

        top_stocks = retained_stocks + new_picks_final

        top_stocks = top_stocks[:Portfolio_N]  # Just in case extra stocks crept in

        inPortfolio.loc[data1.index[end_idx], top_stocks] = 1


        #Factor Score of universe and portfolio
        portfolio_score = bins[top_stocks].mean()
        universe_score = bins.mean()

        # Average forward return of top N stocks
        portfolio_return = forward_return[top_stocks].mean()


        #Turnover, transaction costs
        current_portfolio = set(top_stocks)
        new_stocks = current_portfolio - prev_portfolio
        num_new_stocks = len(new_stocks)
        new_stock_turnover = num_new_stocks / Portfolio_N

        overlapping_stocks = current_portfolio & prev_portfolio
        full_prev_weights = prev_nav / prev_nav.sum() 

        if overlapping_stocks:

            # Extract only overlapping stock weights
            prev_weights = full_prev_weights[list(overlapping_stocks)]

            # Target new equal weights for overlapping stocks
            new_weights = pd.Series(1 / Portfolio_N, index=overlapping_stocks)

            # Resizing turnover
            resize_turnover = ((prev_weights - new_weights).abs().sum())/2
        else:
            resize_turnover = 0

        turnover = new_stock_turnover + resize_turnover
        cost = 2 * turnover * trade_cost_rate
        net_return = portfolio_return - cost

        prev_nav = (1 + data1[top_stocks].iloc[i : i + forward_months]).prod() # Lagged return for next iteration
        prev_portfolio = current_portfolio  # Lagged portfolio for next iteration


        # Append results
        valid_stock_counts.append(len(valid_stocks))  # total valid stocks
        portfolio_stock_counts.append(len(top_stocks))  # size of portfolio (could be < 30)
        momentum_returns.append(portfolio_return)
        momentum_dates.append(data1.index[end_idx])  # label by forward return month
        portfolio_momentum_scores.append(portfolio_score)
        universe_momentum_scores.append(universe_score)
        turnovers.append(turnover)
        transaction_costs.append(cost)
        net_returns.append(net_return)


    # Construct result DataFrame with additional info

    momentum_df = pd.DataFrame({'Date': momentum_dates,'Momentum_Return': momentum_returns,'Transaction_Cost': transaction_costs,'Post_T.Cost_Return': net_returns,'Turnover': turnovers,'Portfolio_N': portfolio_stock_counts,'Filtered_Universe_N': valid_stock_counts,'Portfolio_Mom_Score': portfolio_momentum_scores,'Universe_Mom_Score': universe_momentum_scores})

    momentum_df.set_index('Date', inplace=True)


    full_df = pd.concat([buy_hold_df, momentum_df])
    full_df.sort_index(inplace=True)



    #relaxed_Contraint_Tax_Agnostic_Taxation_Impact

    tax_portfolio = inPortfolio.copy()
    tax_portfolio = tax_portfolio.replace(0, np.nan)
    tax_portfolio = tax_portfolio.where(pd.isna(tax_portfolio), data1[tax_portfolio.columns])




    trade_cycles = []

    # Iterate over each stock
    for stock in tax_portfolio.columns:
        returns = tax_portfolio[stock]
        in_trade = False
        start_date = None
        cum_return = 1.0
        months_held = 0

        for date, value in returns.items():
            if pd.notna(value):
                if not in_trade:
                    in_trade = True
                    start_date = date
                    cum_return = 1 + value
                    months_held = 1
                else:
                    cum_return *= (1 + value)
                    months_held += 1
            else:
                if in_trade:
                    trade_cycles.append({'Stock': stock, 'Open_Date': start_date,'Close_Date': prev_date,'Total_Return': cum_return - 1,'Time_Period_Months': months_held})
                    in_trade = False
                    cum_return = 1.0
                    months_held = 0
            prev_date = date


    tax_portfolio1 = pd.DataFrame(trade_cycles)



    tax_portfolio1["Year_End"] = tax_portfolio1["Close_Date"].apply(lambda x: pd.Timestamp(year=x.year, month=12, day=1))

    tax_portfolio1["LongTerm"] = (tax_portfolio1["Total_Return"] / Portfolio_N) * (tax_portfolio1["Time_Period_Months"] >= 12)

    tax_portfolio1["ShortTerm"] = (tax_portfolio1["Total_Return"] / Portfolio_N) * (tax_portfolio1["Time_Period_Months"] < 12)

    tax_portfolio2 = tax_portfolio1.groupby("Year_End")[["LongTerm", "ShortTerm"]].sum().reset_index()


    conditions = [
        (tax_portfolio2["LongTerm"] >= 0) & (tax_portfolio2["ShortTerm"] >= 0),  # both gains
        (tax_portfolio2["LongTerm"] >= 0) & (tax_portfolio2["ShortTerm"] < 0),   # LT gain, ST loss
        (tax_portfolio2["ShortTerm"] >= 0) & (tax_portfolio2["LongTerm"] < 0),   # ST gain, LT loss
        (tax_portfolio2["LongTerm"] < 0) & (tax_portfolio2["ShortTerm"] < 0)]     # both losses

    choices = [
        tax_portfolio2["LongTerm"] * LTCG + tax_portfolio2["ShortTerm"] * STCG,
        (tax_portfolio2["LongTerm"] + tax_portfolio2["ShortTerm"]) * LTCG,
        (tax_portfolio2["ShortTerm"] + tax_portfolio2["LongTerm"]) * STCG,
        tax_portfolio2["LongTerm"] * LTCG + tax_portfolio2["ShortTerm"] * STCG]

    # Assign calculated tax liability without forcing minimum 0
    tax_portfolio2["tax_liability"] = pd.Series(np.select(conditions, choices, default=0))


    # Step 1: Create a DataFrame from tax_portfolio2 with Year_End as index
    tax_liability_df = tax_portfolio2[["Year_End", "LongTerm","ShortTerm","tax_liability"]].set_index("Year_End")

    # Step 2: Map tax liability to full_df based on index (Date)
    full_df["LongTerm"] = full_df.index.map(tax_liability_df["LongTerm"])
    full_df["ShortTerm"] = full_df.index.map(tax_liability_df["ShortTerm"])
    full_df["Tax_Loss"] = full_df.index.map(tax_liability_df["tax_liability"])

    full_df[["LongTerm", "ShortTerm","Tax_Loss"]] = full_df[["LongTerm", "ShortTerm","Tax_Loss"]].fillna(0)

    full_df["Post_Tax_Return"] = full_df["Post_T.Cost_Return"] - full_df["Tax_Loss"]



    # Column Orders

    cols = list(full_df.columns)
    last_two = cols[-2:]

    for col in last_two:
        cols.remove(col)

    cols[3:3] = last_two
    full_df = full_df[cols]


    relaxed_contraint_df = full_df.copy()

    relaxed_contraint_df.columns = ["RC_" + col for col in relaxed_contraint_df.columns]

    relaxed_constraint_inPortfolio = inPortfolio.copy()





    final_df_combined = pd.concat([ long_only_df , relaxed_contraint_df ], axis=1)




    # Boolean mask where stock is in relaxed constraint but not in long-only portfolio
    exclusive_relaxed = (relaxed_constraint_inPortfolio == 1) & (long_only_inPortfolio != 1)

    # Count the number of such stocks per date
    exclusive_counts = exclusive_relaxed.sum(axis=1)

    # Divide by Portfolio_N to get the fraction
    Weight_Diff = exclusive_counts / Portfolio_N

    final_df_combined["RC_Weight_Diff"] = Weight_Diff



    #Everything before this is calculations



    
    
    # Ensure index is datetime
    final_df_combined.index = pd.to_datetime(final_df_combined.index)

    # Define durations
    end_date = final_df_combined.index.max()
    durations = {
        "1Y": end_date - pd.DateOffset(years=1),
        "3Y": end_date - pd.DateOffset(years=3),
        "5Y": end_date - pd.DateOffset(years=5),
        "10Y": end_date - pd.DateOffset(years=10),
        "15Y": end_date - pd.DateOffset(years=15),
        "Since_Inception": final_df_combined.index.min()}


    # Stats that should NOT be formatted as percentages
    exclude_format = ["Portfolio_N", "Filtered_Universe_N", "Portfolio_Mom_Score", "Universe_Mom_Score"]

    # Columns that need to be multiplied by 12 before formatting (monthly to annual)
    multiply_by_12 = ['Momentum_Return', 'Transaction_Cost', 'Post_T.Cost_Return','Tax_Loss', 'Post_Tax_Return', 'Turnover', 'LongTerm', 'ShortTerm',]

    # Generate comparison_stat_ DataFrames
    for label, start_date in durations.items():
        df_filtered = final_df_combined[final_df_combined.index >= start_date]

        rc_cols = [col for col in df_filtered.columns if col.startswith("RC_")]
        base_cols = [col for col in df_filtered.columns if not col.startswith("RC_") and "RC_" + col in rc_cols]

        # Long Only averages
        long_avg = df_filtered[base_cols].mean()

        # Relaxed Constraint averages
        rc_base_cols = [col for col in rc_cols if col[3:] in base_cols]
        rc_avg = df_filtered[rc_base_cols].mean()
        rc_avg.index = [col[3:] for col in rc_base_cols]

        # Add RC-only stats (like RC_Weight_Diff)
        rc_only_cols = [col for col in rc_cols if col[3:] not in base_cols]
        for col in rc_only_cols:
            stat = col[3:]
            rc_avg.loc[stat] = df_filtered[col].mean()
            long_avg.loc[stat] = 0.0  # default to 0 for long-only if not present

        # Combine into comparison DataFrame
        comparison_df = pd.DataFrame({
            "Long Only": long_avg,
            "Relaxed Constraint": rc_avg})

        # Format output
        for row in comparison_df.index:
            if row in exclude_format:
                comparison_df.loc[row] = comparison_df.loc[row].map("{:.1f}".format)
            else:
                row_multiplied = comparison_df.loc[row].copy()
                if row in multiply_by_12:
                    row_multiplied = row_multiplied * 12
                row_multiplied = (row_multiplied * 100).map("{:.1f}%".format)
                comparison_df.loc[row] = row_multiplied

        # Store as global variable
        globals()[f"comparison_stat_{label.replace(' ', '_')}"] = comparison_df



    #Calculate Stats

    stats_data = final_df_combined[["Momentum_Return","RC_Momentum_Return","Post_T.Cost_Return","RC_Post_T.Cost_Return","Post_Tax_Return","RC_Post_Tax_Return"]]

    stats_data.index = pd.to_datetime(stats_data.index)

    # Define analysis end date and durations
    end_date = stats_data.index.max()
    durations = {
        "1Y": end_date - pd.DateOffset(years=1),
        "3Y": end_date - pd.DateOffset(years=3),
        "5Y": end_date - pd.DateOffset(years=5),
        "10Y": end_date - pd.DateOffset(years=10),
        "15Y": end_date - pd.DateOffset(years=15),
        "Since_Inception": stats_data.index.min()}


    # Function to calculate stats
    def calculate_stats(df):
        ann_return = df.mean() * 12
        total_return = (1 + df).prod() - 1
        years = len(df) / 12
        cagr = (1 + total_return) ** (1 / years) - 1
        ann_std = df.std() * np.sqrt(12)
        return_by_risk = ann_return / ann_std
        nav = (1 + df).cumprod()
        max_dd = (nav / nav.cummax() - 1).min()
        pos_months = (df > 0).sum() / len(df)
        skew_vals = df.apply(skew)
        kurt_vals = df.apply(lambda x: kurtosis(x, fisher=False))  # normal kurtosis

        return pd.DataFrame({
            "Total Return":total_return,
            "CAGR": cagr,
            "Annualized Return": ann_return,
            "Annualized Std": ann_std,
            "Annualized Return / Risk": return_by_risk,
            "Max Drawdown": max_dd,
            "+ve Months": pos_months,
            "Skewness": skew_vals,
            "Kurtosis": kurt_vals}).T


    # Create named DataFrames
    for label, start_date in durations.items():
        filtered = stats_data[stats_data.index >= start_date]

        summary = calculate_stats(filtered)

        # Format the relevant columns
        for row in summary.index:
            if row not in ["Annualized Return / Risk", "Skewness", "Kurtosis"]:
                summary.loc[row] = (summary.loc[row] * 100).map("{:.1f}%".format)
            else:
                summary.loc[row] = summary.loc[row].map("{:.1f}".format)

        globals()[f"stats_summary_{label.replace(' ', '_')}"] = summary
        #globals()[f"stats_summary_{label.replace(' ', '_')}"] = calculate_stats(filtered)


    # Calendar year returns
    calendar_returns = stats_data.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    calendar_returns.index = calendar_returns.index.year
    calendar_returns_df = calendar_returns
    calendar_returns_df = (calendar_returns * 100).round(1).astype(str) + '%'


    # Example: to access the 5Y summary, use: stats_summary_5Y




    # Ensure datetime index
    final_df_combined.index = pd.to_datetime(final_df_combined.index)

    # Define metrics and labels
    metrics = {
        "Momentum_Return": "Returns before T.Costs and Taxes",
        "Post_T.Cost_Return": "Returns after T.Costs but before Taxes",
        "Post_Tax_Return": "Returns after T.Costs and Taxes"}

    # Transition start date
    transition_start = final_df_combined.index[buy_and_hold_months]

    # Loop through each metric
    # Plot all 3 NAV charts side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (metric, label) in enumerate(metrics.items()):
        long_col = metric
        rc_col = f"RC_{metric}"

        nav_long = (1 + final_df_combined[long_col].fillna(0)).cumprod() * 100
        nav_rc = (1 + final_df_combined[rc_col].fillna(0)).cumprod() * 100

        ax = axes[idx]
        ax.plot(nav_long.index, nav_long, 'k--', label="Long Only")
        ax.plot(nav_rc.index, nav_rc, 'b-', label="Relaxed Constraint")
        ax.axvline(x=transition_start, color='red', linestyle='-', label="Start of Transition")

        # Annotate final values
        ax.text(nav_long.index[-1], nav_long.iloc[-1], f"${nav_long.iloc[-1]:.2f}", color='black', fontsize=7)
        ax.text(nav_rc.index[-1], nav_rc.iloc[-1], f"${nav_rc.iloc[-1]:.2f}", color='blue', fontsize=7)

        ax.set_title(f"${100} Invested:\n{label}", fontsize=10)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Portfolio Value ($)", fontsize=10)
        ax.tick_params(axis='both', labelsize=7)  # 👈 This line controls axis number fonts
        ax.grid(True)
        ax.legend(fontsize=7)

    plt.tight_layout()


    st.markdown('---')
    st.markdown('### Output 1: $100 Invested Line Chart (Jan 2010 - Dec 2024)')
    

    st.pyplot(fig)

   




    # Extract Total Return values from stats_summary_15Y
    summary = stats_summary_15Y.copy()

    # Rename columns for clarity
    labels = ["Returns before T.Costs and Taxes", 
              "Returns after T.Costs but before Taxes", 
              "Returns after T.Costs and Taxes"]

    columns = ["Momentum_Return", "Post_T.Cost_Return", "Post_Tax_Return"]
    rc_columns = [f"RC_{col}" for col in columns]

    # Extract values (remove % and convert to float)
    lo_values = [float(summary.loc["Total Return", col].replace('%', '')) for col in columns]
    rc_values = [float(summary.loc["Total Return", col].replace('%', '')) for col in rc_columns]
    improvements = [(rc / lo - 1) * 100 if lo != 0 else 0 for lo, rc in zip(lo_values, rc_values)]

    # Bar settings
    bar_width = 0.25
    x = np.arange(3)  # For 3 groups

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)  #10
    bar_labels = ['Long Only', 'Relaxed Constraint', 'Improvement %']

    for i in range(3):
        axs[i].bar(0, lo_values[i], width=bar_width, color='black', label='Long Only')
        axs[i].bar(1, rc_values[i], width=bar_width, color='blue', label='Relaxed Constraint')
        axs[i].bar(2, improvements[i], width=bar_width, color='green', label='Improvement %')

        axs[i].set_title(labels[i], fontsize=10)
        axs[i].set_xticks([0, 1, 2])
        axs[i].set_xticklabels(bar_labels, rotation=20, fontsize=7)
        axs[i].set_ylabel("Total Return (%)", fontsize=10)
        axs[i].tick_params(axis='both', labelsize=7)  # Match tick label font size
        axs[i].set_yscale("log")

        # Add text labels above bars
        for j, val in enumerate([lo_values[i], rc_values[i], improvements[i]]):
            axs[i].text(j, val * 1.05 if val > 0 else 0.1, f"{val:.1f}%", ha='center', fontsize=7)


    plt.tight_layout(rect=[0, 0, 1, 0.95])

    st.markdown('---')
    st.markdown('### Output 2: Total Return Bar Chart (Jan 2010 - Dec 2024)')
    #st.markdown('---')

    st.pyplot(plt.gcf())






    # Extract Return / Risk values from stats_summary_15Y
    summary = stats_summary_15Y.copy()

    columns = ["Momentum_Return", "Post_T.Cost_Return", "Post_Tax_Return"]
    rc_columns = [f"RC_{col}" for col in columns]

    labels = [
        "Return / Risk: Before T.Costs and Taxes",
        "Return / Risk: After T.Costs but before Taxes",
        "Return / Risk: After T.Costs and Taxes"]

    # Get actual data from df
    lo_values = [float(summary.loc["Annualized Return / Risk", col]) for col in columns]
    rc_values = [float(summary.loc["Annualized Return / Risk", col]) for col in rc_columns]
    improvements = [(rc / lo - 1) * 100 if lo != 0 else 0 for lo, rc in zip(lo_values, rc_values)]

    # Common Y-axis limits
    y_min = min(lo_values + rc_values) * 0.9
    y_max = max(lo_values + rc_values) * 1.15
    imp_min = 0
    imp_max = max(improvements) * 1.25

    # Create plots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    bar_labels = ['Long Only', 'Relaxed Constraint', 'Improvement %']

    # Shared right axis for improvement
    right_axes = [ax.twinx() for ax in axs]
    for ax in right_axes:
        ax.set_ylim(imp_min, imp_max)
        ax.set_yticks([])

    # Only show right y-axis labels for the first subplot
    right_axes[0].set_ylabel("Improvement %", fontsize=9)
    right_axes[0].set_yticks(np.linspace(imp_min, imp_max, 5))
    
    
    # Loop through each subplot
    for i in range(3):
        ax = axs[i]

        # Plot LO and RC bars
        ax.bar(0, lo_values[i], color='black', width=0.25)
        ax.bar(1, rc_values[i], color='blue', width=0.25)
        ax.set_ylim(y_min, y_max)
        ax.set_title(labels[i], fontsize=9)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(bar_labels, rotation=20, fontsize=9)

        # Annotate LO and RC bars
        for j, val in enumerate([lo_values[i], rc_values[i]]):
            ax.text(j, val * 1.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

        # Plot improvement bar on the right axis
        right_axes[i].bar(2, improvements[i], color='green', width=0.25)
        right_axes[i].text(2, improvements[i] * 1.01, f"{improvements[i]:.1f}%", ha='center', va='bottom', fontsize=9)

        # Remove left y-axis ticks/labels for all but first
        if i != 0:
            ax.tick_params(left=False, labelleft=False)
        else:
            ax.set_ylabel("Return / Risk", fontsize=9)



    plt.tight_layout(rect=[0, 0, 1, 0.95])

    st.markdown('---')
    st.markdown('### Output 3: Return / Risk Bar Chart (Jan 2010 - Dec 2024)')
    

    st.pyplot(plt.gcf())






    

    # Extract tax values
    lo_tax = final_df_combined["Tax_Loss"].iloc[buy_and_hold_months - 1] * 100
    rc_tax = final_df_combined["RC_Tax_Loss"].iloc[buy_and_hold_months - 1] * 100
    tax_saving = lo_tax - rc_tax

    # Extract transaction cost values
    lo_tc = final_df_combined["Transaction_Cost"].iloc[buy_and_hold_months] * 100
    rc_tc = final_df_combined["RC_Transaction_Cost"].iloc[buy_and_hold_months] * 100
    tc_saving = lo_tc - rc_tc

    # Bar chart setup
    bar_width = 0.2
    labels1 = ["Long Only", "Relaxed Constraint", "Tax Savings"]
    labels2 = ["Long Only", "Relaxed Constraint", "Transaction Cost Savings"]
    colors = ["black", "blue", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left chart: Taxes ---
    tax_values = [lo_tax, rc_tax, tax_saving]
    bars1 = axes[0].bar(labels1, tax_values, color=colors, width=bar_width)
    axes[0].set_title("Taxes paid in Transition Month", fontsize=10)
    axes[0].set_ylabel("Taxes paid in transition month (%)", fontsize=10)
    axes[0].set_ylim(0, max(tax_values) * 1.25)

    for bar, val in zip(bars1, tax_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val * 1.01, f"{val:.1f}%", ha='center', fontsize=10)

    # --- Right chart: Transaction Costs ---
    tc_values = [lo_tc, rc_tc, tc_saving]
    bars2 = axes[1].bar(labels2, tc_values, color=colors, width=bar_width)
    axes[1].set_title("Transaction Costs in Transition Month", fontsize=10)
    axes[1].set_ylabel("Transaction Costs in transition month (%)", fontsize=10)
    axes[1].set_ylim(0, max(tc_values) * 1.25)

    for bar, val in zip(bars2, tc_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.01, f"{val:.1f}%", ha='center', fontsize=10)

    # Increase space between plots
    plt.subplots_adjust(wspace=30)

    plt.tight_layout()

    st.markdown('---')
    transition_date = final_df_combined.index[buy_and_hold_months-1]
    transition_month_str = transition_date.strftime("%b '%y")
    st.markdown(f"### Output 4: Expenses (Taxes and Transaction Costs) in the Transition Month ({transition_month_str})")


    st.pyplot(plt.gcf())







    # Extract tax values
    lo_tax = final_df_combined["Tax_Loss"].sum() * 100
    rc_tax = final_df_combined["RC_Tax_Loss"].sum() * 100
    tax_saving = lo_tax - rc_tax

    # Extract transaction cost values
    lo_tc = final_df_combined["Transaction_Cost"].sum() * 100
    rc_tc = final_df_combined["RC_Transaction_Cost"].sum() * 100
    tc_saving = lo_tc - rc_tc

    # Bar chart setup
    bar_width = 0.2
    labels1 = ["Long Only", "Relaxed Constraint", "Tax Savings"]
    labels2 = ["Long Only", "Relaxed Constraint", "Transaction Cost Savings"]
    colors = ["black", "blue", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left chart: Taxes ---
    tax_values = [lo_tax, rc_tax, tax_saving]
    bars1 = axes[0].bar(labels1, tax_values, color=colors, width=bar_width)
    axes[0].set_title("Sum of Taxes Paid", fontsize=10)
    axes[0].set_ylabel("Taxes (%)", fontsize=10)
    axes[0].set_ylim(0, max(tax_values) * 1.25)

    for bar, val in zip(bars1, tax_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val * 1.01, f"{val:.1f}%", ha='center', fontsize=10)

    # --- Right chart: Transaction Costs ---
    tc_values = [lo_tc, rc_tc, tc_saving]
    bars2 = axes[1].bar(labels2, tc_values, color=colors, width=bar_width)
    axes[1].set_title("Sum of Transaction Costs", fontsize=10)
    axes[1].set_ylabel("Transaction Costs (%)", fontsize=10)
    axes[1].set_ylim(0, max(tc_values) * 1.25)

    for bar, val in zip(bars2, tc_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.01, f"{val:.1f}%", ha='center', fontsize=10)

    # Increase space between plots
    plt.subplots_adjust(wspace=30)

    plt.tight_layout()

    st.markdown('---')
    st.markdown('### Output 5: Sum of Taxes and Transaction Costs (Jan 2010 - Dec 2024)')
    

    st.pyplot(plt.gcf())






    lo_score = final_df_combined["Portfolio_Mom_Score"].mean()
    rc_score = final_df_combined["RC_Portfolio_Mom_Score"].mean()
    univ_score = final_df_combined["Universe_Mom_Score"].mean()

    values = [lo_score, rc_score, univ_score]
    labels = ["Long Only", "Relaxed Constraint", "Avg. S&P 500"]
    colors = ["black", "blue", "grey"]
    bar_width = 0.2

    # === Transition Month ===
    transition_date = final_df_combined.index[buy_and_hold_months]
    transition_month_str = transition_date.strftime("%b '%y")

    # === Streamlit Header ===
    st.markdown('---')
    st.markdown(f"### Output 6: Momentum Factor Score (out of 10) ({transition_month_str} - Dec '24')")

    # === Create Columns for Side-by-Side Layout ===
    col1, col2 = st.columns([1, 1])

    # === Left Chart: Actual Bar Chart ===
    with col1:
        fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))
        bars = ax1.bar(labels, values, color=colors, width=bar_width)

        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val:.2f}", 
                    ha='center', fontsize=7)

        ax1.set_ylim(0, 10)
        ax1.set_ylabel("Momentum Score", fontsize=7)
        ax1.set_title("Momentum Score (out of 10)", fontsize=7)
        ax1.tick_params(axis='both', labelsize=6)
        fig1.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.25)
        st.pyplot(fig1)

    # === Right Chart: Completely Blank White Chart ===
    with col2:
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
        ax2.axis("off")  # turn off all axis and lines
        fig2.patch.set_facecolor('white')
        st.pyplot(fig2)



    st.markdown('---')
    st.markdown("### Output 7: Summary Statistics and Portfolio Characteristics (Jan 2010 - Dec 2024)")

    # Rename columns in stats_summary_15Y only
    summary_df = stats_summary_15Y.rename(columns={
        "Momentum_Return": "Long Only - Returns before T.Costs and Taxes",
        "Post_T.Cost_Return": "Long Only - Returns after T.Costs but before Taxes",
        "Post_Tax_Return": "Long Only - Returns after T.Costs and Taxes",
        "RC_Momentum_Return": "Relaxed Constraint - Returns before T.Costs and Taxes",
        "RC_Post_T.Cost_Return": "Relaxed Constraint - Returns after T.Costs but before Taxes",
        "RC_Post_Tax_Return": "Relaxed Constraint - Returns after T.Costs and Taxes"})

    # Style both tables
    styled_summary = summary_df.style.set_table_styles([
        {"selector": "th", "props": [("font-weight", "bold"), ("font-size", "14px")]},
        {"selector": "td", "props": [("font-size", "13px")]}])

    styled_comparison = comparison_stat_15Y.style.set_table_styles([
        {"selector": "th", "props": [("font-weight", "bold"), ("font-size", "14px")]},
        {"selector": "td", "props": [("font-size", "13px")]}])

    # Display the first table
    st.markdown("#### A. Summary Statistics")
    st.dataframe(styled_summary, use_container_width=True)

    # Add vertical space between tables
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Display the second table
    st.markdown("#### B. Portfolio Characteristics")
    st.dataframe(styled_comparison, use_container_width=True)
