import numpy as np
import pandas as pd

def parse_utc_to_time_info(utc_string: str) -> dict:
    """
    Convert a UTC datetime string to year, month, week number, day, and 4-hour block.
    
    Parameters:
        utc_string (str): UTC datetime string, e.g. '2025-09-11T15:30:00Z'
    
    Returns:
        dict: {
            'year': int,
            'month': int (1-12),
            'week': int (ISO week number, 1-53),
            'day': int (1-31),
            'block': int (1-6 for 4-hour blocks)
        }
    """
    # Parse string to pandas Timestamp (handles ISO format)
    ts = pd.to_datetime(utc_string, utc=True)
    
    year = ts.year
    month = ts.month
    day = ts.day
    week = ts.isocalendar().week  # ISO week number
    
    # Determine 4-hour block
    hour = ts.hour
    block = (hour // 4) + 1  # 0-3 -> 1, 4-7 -> 2, ..., 20-23 -> 6
    
    return {
        'year': year,
        'month': month,
        'week': week,
        'day': day,
        'block': block
    }

def get_bid_profiles(auction_data: pd.DataFrame, time: dict, volume: float) -> pd.DataFrame:

    # aFFR weekly auction
    if volume > 0:
        aFFR_weekly_data = auction_data.loc[(auction_data['description'] == "Secondary control Auction SRL+") & 
                                    (auction_data['year'] == time['year']) & 
                                    (auction_data['week'] == time['week'])]
    elif volume < 0: 
        aFFR_weekly_data = auction_data.loc[(auction_data['description'] == "Secondary control Auction SRL-") & 
                                    (auction_data['year'] == time['year']) & 
                                    (auction_data['week'] == time['week'])]

    # mFRR weekly auction
    if volume > 0: 
        mFFR_weekly_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['week'] == time['week'])]
    elif volume < 0: 
        mFFR_weekly_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['week'] == time['week'])]

    # mFFR daily auction 
    if volume > 0: 
        if time['block'] == 1:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP 00:00 bis 04:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 2:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP 04:00 bis 08:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 3:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP 08:00 bis 12:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 4:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP 12:00 bis 16:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 5:  
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP 16:00 bis 20:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 6:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power UP 20:00 bis 24:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
    elif volume < 0:
        if time['block'] == 1:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN 00:00 bis 04:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 2:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN 04:00 bis 08:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 3:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN 08:00 bis 12:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 4:
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN 12:00 bis 16:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 5:  
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN 16:00 bis 20:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]
        if time['block'] == 6:  
            mFFR_daily_data = auction_data.loc[(auction_data['description'] == "Tertiary Power DOWN 20:00 bis 24:00") & 
                                        (auction_data['year'] == time['year']) & 
                                        (auction_data['month'] == time['month']) & 
                                        (auction_data['day'] == time['day'])]  

    return aFFR_weekly_data, mFFR_weekly_data, mFFR_daily_data


def get_mFFR_profile(mFFR_weekly_data: pd.DataFrame, mFFR_daily_data: pd.DataFrame) -> pd.DataFrame:
    mFFR_data = pd.concat([mFFR_weekly_data, mFFR_daily_data])
    return mFFR_data

# Test
auction_data_path = "../Data/data_files/df_PRL_SRL_TRL_bids.parquet"
auction_data = pd.read_parquet(auction_data_path)

volume = 100.0

time_utc = "2023-09-11T15:30:00Z"
time = parse_utc_to_time_info(time_utc)
print(time)

aFFR_weekly_data, mFFR_weekly_data, mFFR_daily_data = get_bid_profiles(auction_data, time, volume)
mFFR_data = get_mFFR_profile(mFFR_weekly_data, mFFR_daily_data)