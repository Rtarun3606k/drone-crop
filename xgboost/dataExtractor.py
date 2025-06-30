import os
import csv
import json
import requests
from datetime import datetime
import pandas as pd

# Define the input CSV file path
input_csv_path = "/home/dragoon/coding/drone-crop/xgboost/UAVSoyabean_Mosaic.csv"
output_csv_path = "/home/dragoon/coding/drone-crop/xgboost/UAVSoyabean_Mosaic_Enhanced.csv"

# Define location information (assuming these are the locations where images were taken)
locations = {
    "Chikhali": {"longitude": 74.38506666, "latitude": 17.33345, "district": "Sangli"},
    "Upavale": {"longitude": 74.0937688, "latitude": 16.9732274, "district": "Sangli"},
    "Hingangaon": {"longitude": 74.3453593, "latitude": 17.26637, "district": "Sangli"},
    "Kadepur": {"longitude": 74.364642, "latitude": 17.293703, "district": "Sangli"},
    "Banawadi": {"longitude": 74.1943023, "latitude": 17.3179823, "district": "Satara"},
    "Goware": {"longitude": 74.208238, "latitude": 17.286995, "district": "Satara"}
}

# Use Banawadi as the default location (you can change this or implement a better method to determine location)
default_location = "Banawadi"

# Function to fetch weather data for a date and location
def fetch_weather_data(date_str, latitude, longitude):
    """Fetch weather data for a specific date and location coordinates"""
    # Format date as YYYY-MM-DD
    date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={formatted_date}&end_date={formatted_date}&hourly=cloudcover,precipitation,relative_humidity_2m,windspeed_10m,weathercode"
    
    try:
        response = requests.get(url)
        print(f"Fetching weather data from: {url}")
        if response.status_code == 200:
            print(f"Successfully fetched weather data for {formatted_date} at ({latitude}, {longitude})")
            data = response.json()
            return data
        else:
            print(f"Error fetching weather data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception when fetching weather data: {e}")
        return None

# Function to get weather data for a specific hour
def get_hour_weather(weather_data, target_hour):
    """Extract weather data for a specific hour from the API response"""
    if not weather_data or "hourly" not in weather_data:
        return None
    
    times = weather_data["hourly"]["time"]
    for i, time_str in enumerate(times):
        hour = int(time_str.split("T")[1].split(":")[0])
        if hour == target_hour:
            return {
                "cloudcover": weather_data["hourly"]["cloudcover"][i],
                "precipitation": weather_data["hourly"]["precipitation"][i],
                "humidity": weather_data["hourly"]["relative_humidity_2m"][i],
                "windspeed": weather_data["hourly"]["windspeed_10m"][i],
                "weathercode": weather_data["hourly"]["weathercode"][i]
            }
    
    return None

# Weather code mapping
weather_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast/Cloudy",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

# Read the input CSV file
try:
    df = pd.read_csv(input_csv_path)
    print(f"Successfully read {len(df)} rows from {input_csv_path}")
except Exception as e:
    print(f"Error reading input CSV: {e}")
    exit(1)

# Add new columns for the enhanced data
df['Location'] = default_location
df['District'] = locations[default_location]['district']
df['Latitude'] = locations[default_location]['latitude']
df['Longitude'] = locations[default_location]['longitude']
df['Cloudcover (%)'] = None
df['Precipitation (mm)'] = None
df['Humidity (%)'] = None
df['Windspeed (km/h)'] = None
df['Weathercode'] = None
df['Weather Description'] = None
df['Image Type'] = 'Mosaic'  # Assuming all images are of type Mosaic based on filename

# Cache for weather data to avoid redundant API calls
weather_cache = {}

# Process each row
for index, row in df.iterrows():
    try:
        # Get the parsed date/time from the row
        date_time_str = str(row['Parsed Date/Time'])
        
        if pd.notna(date_time_str):
            # Extract date and hour
            date_obj = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            date_str = date_obj.strftime("%Y-%m-%d")
            hour = date_obj.hour
            
            # Check if we already have weather data for this date
            if date_str not in weather_cache:
                # Fetch weather data for the location and date
                weather_data = fetch_weather_data(
                    date_str, 
                    locations[default_location]['latitude'], 
                    locations[default_location]['longitude']
                )
                weather_cache[date_str] = weather_data
            
            # Get weather for the specific hour (using the hour from the image timestamp)
            hour_weather = get_hour_weather(weather_cache[date_str], hour)
            
            if hour_weather:
                df.at[index, 'Cloudcover (%)'] = hour_weather['cloudcover']
                df.at[index, 'Precipitation (mm)'] = hour_weather['precipitation']
                df.at[index, 'Humidity (%)'] = hour_weather['humidity']
                df.at[index, 'Windspeed (km/h)'] = hour_weather['windspeed']
                df.at[index, 'Weathercode'] = hour_weather['weathercode']
                df.at[index, 'Weather Description'] = weather_codes.get(
                    hour_weather['weathercode'],
                    f"Unknown code: {hour_weather['weathercode']}"
                )
                print(f"Added weather data for {row['Filename']} ({date_str} {hour}:00)")
            else:
                print(f"No weather data found for {row['Filename']} ({date_str} {hour}:00)")
    except Exception as e:
        print(f"Error processing row {index} ({row['Filename'] if 'Filename' in row else 'unknown'}): {e}")

# Extract additional features from filename
df['Flight ID'] = df['Filename'].apply(lambda x: x.split('_')[2] if len(x.split('_')) > 2 else 'Unknown')
df['Sequence Number'] = df['Filename'].apply(lambda x: x.split('_')[-1].split('.')[0] if len(x.split('_')) > 3 else 'Unknown')

# Write the enhanced data to a new CSV file
try:
    df.to_csv(output_csv_path, index=False)
    print(f"\nEnhanced metadata successfully written to: {output_csv_path}")
    print(f"Added columns: Location, District, Latitude, Longitude, Weather data, Image Type, Flight ID, Sequence Number")
except Exception as e:
    print(f"Error writing output CSV: {e}")

print("\nProcess completed.")