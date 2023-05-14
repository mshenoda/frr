import pandas as pd
import random

def random_sample_csv(input_file, output_file):
    # Read the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Drop rows with rating equal to 0
    df = df[df['rating'] != 0]
    
    # Calculate the number of rows to sample (20% of the total)
    sample_size = int(len(df) * 0.5)
    
    # Randomly sample the DataFrame
    sampled_df = df.sample(n=sample_size, random_state=random.seed())
    
    # Save the sampled data to a new CSV file
    sampled_df.to_csv(output_file, index=False)