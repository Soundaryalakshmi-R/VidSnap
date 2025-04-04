import pandas as pd
import numpy as np
anno_df = pd.read_csv('/Users/apple/Desktop/datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv', sep='\t')
print("Annotations:")
print(anno_df.head())
print(anno_df.columns)

# Load metadata
info_df = pd.read_csv('/Users/apple/Desktop/datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv', sep='\t')
print("\nMetadata:")
print(info_df.head())
print(info_df.columns)