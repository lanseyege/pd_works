import pandas as pd
import os, sys
import numpy as np

file_name = "/mnt/nfs_a/cvat-data/zhw/batch3_right/data/chunk-000/episode_000000.parquet"

df = pd.read_parquet(file_name, engine="pyarrow")
print(df.head())
print(df.info())
print(df.describe())
print()

print(df["action"].head().iloc[3])
'''
print(df["observation.state"].iloc[20])
print(df["observation.state"].iloc[60])
print(df["observation.state"].iloc[100])
print(df["observation.state"].iloc[130])
print(df["observation.state"].iloc[160])
print(df["observation.state"].iloc[190])
'''
depth = np.array(df['front_top_depth_raw'].iloc[9])
print(depth.shape)
#print(depth[0].shape)
#print(depth[0][0].shape)
#print(depth)
