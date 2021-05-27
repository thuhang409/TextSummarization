import config
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import DataClean

class MakeDatasetSummary:
  def __init__(self):
    pass

  def LoadData(self, path):
    df = pd.read_csv(path)
    return df

  def FilterData(self, df, thresh = 0.75, text_col = 'Text', summary_col = 'Summary'):
    '''
    Drop another columns (except Summary and Text)
    Drop data point has length of Summary too long (bigger than thresh of length of Text) 
    '''
    df = df[df['Summary'].str.len() < 0.75*df['Text'].str.len()]
    df = df[['Summary', 'Text']]
    return df
  
  def SplitData(self, df, train_size = config.train_size, random_state = 0, reset_index = True):
    '''
      Split DataFrame to train dataframe and test dataframe
    '''
    train_data, test_data = train_test_split(df, train_size=train_size, random_state = random_state)
    if reset_index:
      print('reset')
      train_data = train_data.reset_index(drop = True)[['Text','Summary']]
      test_data = test_data.reset_index(drop = True)[['Text','Summary']]
    return train_data, test_data

  def SaveData(self, df, path_save):
    df.to_csv(path_save, index=False)
    print("Saved!")


if __name__ == "__main__":
  makedata = MakeDatasetSummary()
  df = makedata.LoadData(config.data_path)
  df = makedata.FilterData(df)
  if config.clean_data:
    clean = DataClean()
    # ds_cleaned = pd.DataFrame(columns=['headline','text'])
    df['Text'] = df['Text'].apply(lambda x: clean.Text_cleaned(x)) 
    df['Summary'] = df['Summary'].apply(lambda x: clean.Summary_cleaned(x))
  df_train, df_test = makedata.SplitData(df, reset_index = True)
  if config.save_data:
    makedata.SaveData(df_train, config.data_path_save + 'train.csv')
    makedata.SaveData(df_test, config.data_path_save + 'test.csv')