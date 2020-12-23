import pandas as pd
from datetime import datetime

# Load data and subset to relevant columns
activities = pd.read_csv('./data/strava_activities.csv')
activities = activities[['id', 'start_date_local', 'type', 'distance', 'moving_time', 'total_elevation_gain']]

# Create year and week variable with weeks starting on Monday
activities['start_date_local'] = pd.to_datetime(activities['start_date_local'])
activities['monday'] = activities['start_date_local'] -  pd.to_timedelta(arg=activities['start_date_local'].dt.weekday, unit='D')
activities['monday'] = activities['monday'].dt.strftime('%m-%d-%Y')

# Convert active seconds to active hours
activities['moving_time'] = activities['moving_time']/360

# Modify groups:
activities['type'] = activities['type'].replace({
    'Elliptical':'Other',
    'Workout':'Other',
    'Walk':'Other',
    'Hike':'Other',
    'WeightTraining':'Other'
})

# Collapse by week and activity type
#   1. distance: meters
#   2. moving_time: seconds
#   3. total_elevation_gain: meters
activities = activities.groupby(['monday', 'type'], as_index=False).agg({
    'distance':'sum',
    'moving_time':'sum',
    'total_elevation_gain':'sum',
})

# Reshape wide on activity type
activities = activities.pivot_table(
    index='monday',
    columns='type',
    values=['distance', 'moving_time', 'total_elevation_gain'],
    fill_value = 0
).reset_index()

# Collapse multi-index of header columns into one row

# distance  |  distance        ->    Hike_distance  |  Other_distance
# Hike      |  Other

new_cols = [('{1}_{0}'.format(*tup)) for tup in activities.columns]
activities.columns = new_cols
activities = activities.rename({'_monday': 'monday'}, axis='columns')

# Sort on Monday
activities['monday'] = pd.to_datetime(activities['monday'])
activities = activities.sort_values(by='monday')
activities['monday'] = activities['monday'].dt.strftime('%m-%d-%Y')

# Write data
activities.to_csv('./data/strava_activities_sub.csv', index=False)
