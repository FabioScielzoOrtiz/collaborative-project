import polars as pl
import seaborn as sns

data_path = r'C:\Users\fscielzo\Documents\Videos-Projects\Collaborative-Projects-GitHub\user1_project\data\madrid_houses_processed.csv'
madrid_houses_df = pl.read_csv(data_path)
sns.histplot(data=madrid_houses_df, x='buy_price')
