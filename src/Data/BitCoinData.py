import pandas as pd


pd.set_option('display.max_columns', None)


df = pd.read_csv(r"D:\College Projects\BitCoin Trading Agent\src\Data\TrainingData_with_indicators.csv")



print(df.tail())
