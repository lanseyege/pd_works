import pandas as pd
import os, sys


'''
file_name = "/mnt/nfs_a/cvat-data/zhw/batch3_right/data/chunk-000/episode_000000.parquet"

df = pd.read_parquet(file_name, engine="pyarrow")
print(df.head())
print(df.info())
#print(df.describe())
#print()
'''

file_path = "/mnt/nfs_a/cvat-data/zhw/batch3_right/data/chunk-000/"
new_path = "/home/yuanye/dataset/batch3_right/data/chunk-000/"
file_names = os.listdir(file_path)
#print(file_names)
i = 0
for file_name in file_names:
    fp = file_path + file_name
    df = pd.read_parquet(fp, engine="pyarrow")
    df_dropped = df.drop('front_top_depth_raw', axis=1)
    df_dropped.to_parquet(new_path + file_name, engine='pyarrow', index=False)
    i += 1
print(f"{i} files written!!")
