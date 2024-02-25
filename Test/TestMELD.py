import unittest
import pandas as pd
import os
import glob

# Below we have the class TestMELD which is used to test  the MELD dataset
class TestMELD(unittest.TestCase):
    # test to check to check if the folder path to export exists
    def test_folder_path(self):
        self.assertTrue(os.path.exists(self.folder))

    # test to check if the train dataframe is not empty
    def test_train_df_not_empty(self):
        self.assertFalse(self.train_df.empty)
    
    # test to check if the dev dataframe is not empty
    def test_dev_df_not_empty(self):
        self.assertFalse(self.dev_df.empty)

    # test to check if the test dataframe is not empty
    def test_test_df_not_empty(self):
        self.assertFalse(self.test_df.empty)

    # test to check if the train dataframe has the correct number of columns
    def test_train_df_columns(self):
        self.assertEqual(len(self.train_df.columns), 12)
    
    # test to check if the dev dataframe has the correct number of columns
    def test_dev_df_columns(self):
        self.assertEqual(len(self.dev_df.columns), 12)

    # test to check if the test dataframe has the correct number of columns
    def test_test_df_columns(self):
        self.assertEqual(len(self.test_df.columns), 12)

    # test to check if the train dataframe has the correct number of rows
    def test_train_df_rows(self):
        self.assertEqual(len(self.train_df), 9988)

    # test to check if the dev dataframe has the correct number of rows
    def test_dev_df_rows(self):
        self.assertEqual(len(self.dev_df), 1108)
    
    # test to check if the test dataframe has the correct number of rows
    def test_test_df_rows(self):
        self.assertEqual(len(self.test_df), 2610)

    # test to check if the df has the correct number of emotions
    def test_df_emotions(self):
        self.assertEqual(len(self.df['Emotion'].unique()), 7)

    # test to check if the df has the correct number of sentiments
    def test_df_sentiments(self):
        self.assertEqual(len(self.df['Sentiment'].unique()), 3)

    # test to check if the df has the correct number of data
    def test_df_data(self):
        self.assertEqual(len(self.df['Data'].unique()), 3)
    
    # test to check if all the filenames are unique
    def test_df_filename_unique(self):
        self.assertEqual(len(set(zip(self.df['Data'],self.df['filename']))), len(self.df))

    # test to check the datatype for all the columns in the df
    def test_df_datatype(self):
        self.assertEqual(self.df['Sr No.'].dtype, 'int64')
        self.assertEqual(self.df['Utterance'].dtype, 'object')
        self.assertEqual(self.df['Speaker'].dtype, 'object')
        self.assertEqual(self.df['Emotion'].dtype, 'object')
        self.assertEqual(self.df['Sentiment'].dtype, 'object')
        self.assertEqual(self.df['Dialogue_ID'].dtype, 'int64')
        self.assertEqual(self.df['Utterance_ID'].dtype, 'int64')
        self.assertEqual(self.df['Season'].dtype, 'int64')
        self.assertEqual(self.df['Episode'].dtype, 'int64')
        self.assertEqual(self.df['StartTime'].dtype, 'object')
        self.assertEqual(self.df['EndTime'].dtype, 'object')
        self.assertEqual(self.df['Data'].dtype, 'object')
        self.assertEqual(self.df['filename'].dtype, 'object')
        self.assertEqual(self.df['filepath'].dtype, 'object')

    # test to check if there are no duplicate filenames in the df
    def test_df_filename_duplicate(self):
        self.assertEqual(len(self.df['filename'].unique()), len(self.df))

    # test to check if the number of wav files is equal to the number of rows in the df
    def test_df_wav_rows(self):
        wav_count = len(glob.glob(self.folder + '/**/*.wav', recursive=True))
        self.assertEqual(wav_count, len(self.df))

if __name__ == '__main__':
    unittest.main()