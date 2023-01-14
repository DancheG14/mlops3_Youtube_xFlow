import pandas as pd
 
df = pd.read_csv('/data.csv', header=None)
 
df[0] = (df[0]-df[0].min())/(df[0].max()-df[0].min())
 
with open('/data_processed.csv', 'w') as f:
    for i, item in enumerate(df[0].values):
        f.write("{},{}\n".format(i, item))