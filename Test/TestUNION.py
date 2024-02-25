# in IEMOCAP_test.py

import unittest
import pandas as pd
import os
# import IEMOCAP_data as IEMOCAPDataProcessor  # Import your data processing module

class TestUNION(unittest.TestCase):
    # # test if the folder path exists
    # def test_folder_path(self):
    #     self.assertTrue(os.path.exists(self.folder))

    # test if the number of rows in iemocap is correct
    def test_iemocap_rows(self):
        self.assertEqual(len(self.iemocap_df), 10039)

    # test if the number of rows in meld is correct
    def test_meld_rows(self):
        self.assertEqual(len(self.meld_df), 13706)

    # test if the number of rows in union is correct
    def test_df_rows(self):
        self.assertEqual(len(self.df), 20157)

    # test if the number of columns in union is correct
    def test_df_columns(self):
        self.assertEqual(len(self.df.columns), 11)

    # test if the number of emotions in the union is correct
    def test_df_emotions_length(self):
        self.assertEqual(len(self.df['emotion_label'].unique()), len(self.emotions))

    # test if the clean_text column has any new line, carriage return or tab characters
    def test_df_clean_text_nline_tab(self):
        self.assertFalse(self.df['clean_text'].str.contains('\n').any())
        self.assertFalse(self.df['clean_text'].str.contains('\r').any())
        self.assertFalse(self.df['clean_text'].str.contains('\t').any())

    #test if the clean_text column has any urls
    def test_df_clean_text_urls(self):
        self.assertFalse(self.df['clean_text'].str.contains('http').any())

    # test if the clean_text column has any double spaces
    def test_df_clean_text_double_spaces(self):
        self.assertFalse(self.df['clean_text'].str.contains('  ').any())

    # test if the asr_clean_text column has any new line, carriage return or tab characters
    def test_df_asr_clean_nline_tab(self):
        self.assertFalse(self.df['asr_clean_text'].str.contains('\n').any())
        self.assertFalse(self.df['asr_clean_text'].str.contains('\r').any())
        self.assertFalse(self.df['asr_clean_text'].str.contains('\t').any())

    #test if the asr_clean_text column has any urls
    def test_df_asr_clean_text_urls(self):
        self.assertFalse(self.df['asr_clean_text'].str.contains('http').any())

    # test if any column has null values
    def test_df_null(self):
        self.assertFalse(self.df.isnull().values.any())
    
    # test that the emotions in the dataframe match the emotion_num dictionary
    def test_df_emotion_values(self):
        self.assertEqual(self.df['emotion_label'].sort_values().unique().tolist(), sorted(self.emotions))

    # test if the number of rows in the train_df
    def test_train_row_count(self):
        self.assertEqual(len(self.train_df), 16125)

    # test the number of columns in the train_df
    def test_train_column_count(self):
        self.assertEqual(len(self.train_df.columns), 11)

    # test if the number of rows in the val_df
    def test_val_row_count(self):
        self.assertEqual(len(self.val_df), 2016)

    # test if the number of columns in the val_df
    def test_val_column_count(self):
        self.assertEqual(len(self.val_df.columns), 11)

    # test if the number of rows in the test_df
    def test_test_rows_count(self):
        self.assertEqual(len(self.test_df), 2016)
    
    # test if the number of columns in the test_df
    def test_test_column_count(self):
        self.assertEqual(len(self.test_df.columns), 11)

if __name__ == '__main__':
    unittest.main()
