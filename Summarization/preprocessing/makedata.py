import pandas as pd
from sklearn.model_selection import train_test_split

class MakeDataset:
  def __init__(self):
    pass

  def LoadData(self, path):
    '''
    Load dataset từ path
    '''
    df = pd.read_csv(path)
    df.Summary = df.Summary.astype(str)
    df.Text = df.Text.astype(str)
    return df

  def FilterData(self, df, thresh = 0.75, text_col = 'Text', summary_col = 'Summary'):
    '''
    Xóa những cột khác, chỉ để lại Text và Summary
    Chọn những điểm dữ liệu tốt - có Summary ngắn hơn 0.75 so với Text
    '''
    df = df[df['Summary'].str.len() < thresh * df['Text'].str.len()]
    df = df[['Summary', 'Text']]
    return df
  
  def SplitData(self, df, train_size, reduce_data, rd_train_size, rd_test_size, random_state = 0):
    '''
      Chia tập dữ liệu thành train và test, và giảm số điểm dữ liệu nếu muốn
    '''
    train_data, test_data = train_test_split(df, train_size=train_size, random_state = random_state)
    train_data = train_data.reset_index(drop = True)[['Text','Summary']]
    test_data = test_data.reset_index(drop = True)[['Text','Summary']]

    if reduce_data:
      train_data = train_data[0:rd_train_size]
      test_data = test_data[0:rd_test_size]

    return train_data, test_data


  def SaveData(self, df, path):
    df.to_csv(path, index=False)
    print("Saved!")
