# data/split_fer2013.py
import pandas as pd
df = pd.read_csv('C:/Users/pugli/Downloads/archive(2)/fer2013/fer2013/fer2013.csv')

df[df['Usage']=='Training']   .to_csv('data/train.csv', index=False)
df[df['Usage']=='PublicTest']  .to_csv('data/val.csv',   index=False)
df[df['Usage']=='PrivateTest'] .to_csv('data/test.csv',  index=False)

