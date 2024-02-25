# in IEMOCAP_test.py

import unittest
import pandas as pd
import os
# import IEMOCAP_data as IEMOCAPDataProcessor  # Import your data processing module

# Below we have the class TestIEMOCAP which is used to test the IEMOCAP dataset
class TestIEMOCAP(unittest.TestCase):
    
    # test to check if the folder path exists
    def test_folder_path(self):
        self.assertTrue(os.path.exists(self.folder))

    # test to check the length of the files matches the length of the emotions
    def test_length(self):
        self.assertEqual(len(self.files), len(self.emotions))

    # test to check if the df is a dataframe
    def test_is_dataframe(self):
        self.assertIsInstance(self.df, pd.DataFrame)

    # test to check get filepaths function shape
    def test_df_shape(self):
        self.assertEqual(self.df.shape, (10039, 9))
    
    # test to check if the df emotion is of length 11 (all emotions)
    def test_emotion_length(self):
        self.assertEqual(len(self.df['emotion'].unique()), 11)

    # test to check if the df session column is of length 5 (all sessions)
    def test_session_length(self):
        self.assertEqual(len(self.df['session'].unique()), 5)

    # test to check if the df gender is of length 2
    def test_gender_length(self):
        self.assertEqual(len(self.df['gender'].unique()), 2)

    # test to check if the df method is of length 2
    def test_method_length(self):
        self.assertEqual(len(self.df['method'].unique()), 2)

    # test to check for null values
    def test_null_values(self):
        self.assertFalse(self.df.isnull().any().any())

    # test to check all the other columns are of type object
    def test_columns_datatypes(self):
        self.assertEqual(self.df['filename'].dtype, 'object')
        self.assertEqual(self.df['filepath'].dtype, 'object')
        self.assertEqual(self.df['emotion'].dtype, 'object')
        self.assertEqual(self.df['transcription'].dtype, 'object')
        self.assertEqual(self.df['dataset'].dtype, 'object')
        self.assertEqual(self.df['emotion_label'].dtype, 'object')
        self.assertEqual(self.df['gender'].dtype, 'object')
        self.assertEqual(self.df['method'].dtype, 'object')
        self.assertEqual(self.df['session'].dtype, 'int64')

if __name__ == '__main__':
    unittest.main()
