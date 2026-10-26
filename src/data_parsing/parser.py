"""
parser.py

parser combines multiple CSV files into one, 
keeping only the specified required attributes
and dropping rows that are missing any of those attributes.
"""

import pandas as pd

def combine_and_clean_csvs(file1_path, file2_path, output_path):
    # List of required attributes
    required_attributes = [
        'resort', 
        'season', 
        'latitude', 
        'longitude', 
        'base_elevation_ft', 
        'top_elevation_ft', 
        'elevation_difference_ft', 
        'opening_date', 
        'closing_date', 
        'season_length', 
        'season_snowfall_in'
    ]
    
    # Read the two CSV files
    print(f"Reading {file1_path} and {file2_path}...")
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Concatenate the dataframes vertically
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Keep only the specified required attributes
    combined_df = combined_df[required_attributes]
    
    # Drop rows that are missing any of the required attributes
    cleaned_df = combined_df.dropna(subset=required_attributes, how='any')
    
    # Save the cleaned data to the output CSV file
    cleaned_df.to_csv(output_path, index=False)
    print(f"Combined and cleaned file successfully saved to '{output_path}'. Total valid rows: {len(cleaned_df)}")

if __name__ == "__main__":
    # Configure your input file names and desired output file name here
    file1 = 'data/processed/ski_resort_seasons_complete.csv'
    file2 = 'data/processed/loveland_seasons_history.csv'
    output_file = 'data/processed/all_resort_data_combined.csv'
    
    combine_and_clean_csvs(file1, file2, output_file)