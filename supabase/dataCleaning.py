import pandas as pd
import numpy as np
from typing import List

class DataCleaner:
    def __init__(self, string_cols: List[str] = [], int_cols: List[str] = []):
        self.string_cols: List[str] = string_cols
        self.int_cols: List[str] = int_cols

    def clean_string(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        1. check specific columns in the dataframe, if they exist, convert them to string type.
        """
        for col in self.string_cols:
            if col in data.columns:
                data[col] = data[col].astype("string")

        return data
    
    def clean_integer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        1. check specific columns in the dataframe, if they exist, convert them to integer type.
        """
        for col in self.int_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data
    
    def run_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:

        if self.string_cols != []:
            print("processing string columns...")
            data = self.clean_string(data)
        
        if self.int_cols != []:
            print("processing integer columns...")
            data = self.clean_integer(data)

        return data

if __name__ == "__main__":
    # 定義你的欄位
    str_fields = ['question', 'option1', 'option2', 'option3', 'option4', 'explanation', 'questionType', 'animalType']
    int_fields = ['questionID', 'answer']

    cleaner = DataCleaner(string_cols=str_fields, int_cols=int_fields)

    # 讀取原始資料
    raw_data = pd.read_csv("supabase/database/AnimalQA.csv")
    print("原始資料：")
    print(raw_data.info())

    cleaned_df = cleaner.run_cleaning(raw_data)
    print("清理後資料：")
    print(cleaned_df.info())