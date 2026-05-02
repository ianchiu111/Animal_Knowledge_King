
"""
This supabase client disable RLS (Row Level Security) for convenience during development. 
"""

import os
import pandas as pd
from typing import Dict
from supabase import create_client

from dataCleaning import DataCleaner

from dotenv import load_dotenv
load_dotenv(".env")

class SupabaseClient:
    def __init__(self):
        
        self.url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.key = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
        self.client = create_client(self.url, self.key)

        # === Animal QA ===
        self.animalQA_table = os.getenv("SUPABASE_ANIMALQA_TABLE_NAME")
        self.animalQA_csv = "supabase/database/AnimalQA.csv"

        self.animalQA_string_cols = ['question', 'option1', 'option2', 'option3', 'option4', 'explanation', 'questionType', 'animalType']
        self.animalQA_int_cols = ['questionID', 'answer']
        self.animalQA_target_cols = self.animalQA_string_cols + self.animalQA_int_cols

        self.animalQA_data_cleaner = DataCleaner(
            string_cols=self.animalQA_string_cols,
            int_cols=self.animalQA_int_cols
        )

    def _cols_check(self, data: pd.DataFrame, required_cols: list) -> bool:
        """
        Check if the required columns in table are matched with the columns in the dataframe.
        >>> data: the dataframe to be inserted into database
        >>> required_cols: the columns required by the database table  
        """
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Error: Missing columns in the dataframe: {missing_cols}. Aborting data insertion.")
            return False
        print("Column check passed. All required columns are present in the dataframe.")
        return True
        
    def insert_data(self, table_name: str, data: Dict):
        """
        Insert a single row of data into the specified table in database.
        """

        return self.client.table(table_name).insert(data).execute()

    def update_animalQA_data(self):
        """
        Load csv -> Clean data -> Insert data into database
        - Table name: self.animalQA_table
        - Primary key: questionID
        - Data Schema:
            - questionID: integer (unique identifier for each question)
            - question: string
            - option1: string
            - option2: string
            - option3: string
            - option4: string
            - answer: integer (1, 2, 3, or 4)
            - explanation: string
            - questionType: string
            - animalType: string (cat, dog, general, etc. related to vit model)
        
        Warning
        1. This function will insert all data from the csv file into the database, so it's recommended to use it only when you want to update the entire dataset. 
        2. If the dataframe is not cleaned properly, it may cause errors during insertion, so it's recommended to add some error handling mechanism to catch and log any issues.
        """
        # Load and clean data
        df = pd.read_csv(self.animalQA_csv)
        cleaned_df = self.animalQA_data_cleaner.run_cleaning(data = df)
        cleaned_df = cleaned_df[self.animalQA_target_cols]

        cols_check = self._cols_check(cleaned_df, self.animalQA_target_cols)
        if not cols_check:
            return None

        # Transfer from dataframe to list of dictionary
        data_to_insert = cleaned_df.to_dict(orient='records')

        # Insert data into database
        try:
            self.client.table(self.animalQA_table).upsert(data_to_insert, on_conflict="questionID").execute()
            print(f"Successfully inserted {len(data_to_insert)} records!")
        
        except Exception as e:
            print(f"Error occurred while inserting data: {e}")
            return None
        
    def query_data(self, table_name, columns="*"):
        response = self.client.table(table_name).select(columns).execute()
        return response.data

if __name__ == "__main__":

    supabase_client = SupabaseClient()
    supabase_client.update_animalQA_data()
