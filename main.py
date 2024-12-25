import numpy as np
import pandas as pd
from sklearn.preprocessing import *

class DataProcessor:
    def __init__(self,data):
        """Initialize with the given path of the dataset."""
        self.data=data

    def reading_files(self):
        """
        Reads a file and returns its content as a Pandas DataFrame.
        Supports CSV, Excel (.xlsx, .xls), and JSON (.json) file formats.
        """
        try:

            file_extension = self.data.split('.')[-1].lower()

            if file_extension == 'csv':
                Csv_File = pd.read_csv(self.data)
                return Csv_File
            elif file_extension == 'xlsx' or file_extension == 'xls':
                Excel_File = pd.read_excel(self.data)
                return Excel_File
            elif file_extension == 'josn':
                Josn_file = pd.read_json(self.data)
                return Josn_file
            else:
                print(f'file format .{file_extension} unfortunately not available')
        except FileNotFoundError:
            print("File not found. Please check the file path.")
        except:
            print(' invalid input .')

    def Data_Summary(self):
        """Print key statistical summaries of the dataset."""
        data_summary=self.reading_files()
        try:
            for i in data_summary.columns:
                if pd.api.types.is_numeric_dtype(data_summary[i]):
                    mean_value = np.mean(data_summary[i])
                    median_value = np.median(data_summary[i])
                    std_dev = np.std(data_summary[i])
                    print(f"  Mean: {mean_value}")
                    print(f"  Median: {median_value}")
                    print(f"  Standard Deviation: {std_dev}")
                    if data_summary[i].isnull().all():
                        print(f"Column['{i}'] contains only NaN values.")
                    else:
                        mode_value = np.array(data_summary[i].mode())
                        if len(mode_value) == len(data_summary[i]):
                            print("  Most Frequent Value: all values unique")
                        else:
                            print(f"  Most Frequent Value: {mode_value}")
                    print('-'*10)

                elif pd.api.types.is_string_dtype(data_summary[i]):
                    unique_values = data_summary[i].nunique()
                    if data_summary[i].isnull().all():
                        print(f"Column['{i}'] contains only NaN values.")
                    else:
                        mode_value = np.array(data_summary[i].mode())
                        if len(mode_value) == len(data_summary[i]):
                            print("  Most Frequent Value: all values unique")
                        else:
                            print(f"  Most Frequent Value: {mode_value}")
                    print(f"  Unique Values Count: {unique_values}")
                    print('-' * 10)
        except Exception as e:
            print(f"An error occurred: {e}")

    def Handling_Missing_Values(self):
        """Handle missing values by removing or imputing."""
        handl_data=self.reading_files()
        result = input('choose by the number (1 or 2) : \n 1-remove missing value. \n 2-impute missing value. \n')
        has_null = False
        if result == '1':
            if handl_data.isna().any().any():
                has_null = True
                handl_data.dropna(inplace=True)
        elif result == '2':
            impute_strategy = input("Choose an imputation strategy:\n"
                                    "a. Replace with mean \n"
                                    "b. Replace with median \n"
                                    "c. Replace with mode \n"
                                    "d. Replace with a custom value\n"
                                    "['Note -> a,b and c are available only for numeric columns .\n"
                                    "Choose an option (a, b, c or d): ")
            for i in handl_data.columns:
                if handl_data[i].isna().any():
                    has_null = True
                    if impute_strategy == 'a' and pd.api.types.is_numeric_dtype(handl_data[i]):
                        handl_data[i].fillna(np.mean(handl_data[i]), inplace=True)
                    elif impute_strategy == 'b' and pd.api.types.is_numeric_dtype(handl_data[i]):
                        handl_data[i].fillna(np.median(handl_data[i]), inplace=True)
                    elif impute_strategy == 'c':
                        handl_data[i].fillna(handl_data[i].mode().iloc[0], inplace=True)
                    elif impute_strategy == 'd':
                        fill_value = input(f"Enter a value to replace null values in column['{i}']")
                        fill_value = type(handl_data[i].dropna().iloc[0])(fill_value)
                        handl_data[i].fillna(fill_value, inplace=True)
                    else:
                        return f" invalid input ."
        else:
            return 'invalid input .'

        if not has_null:
            print('Your DataFrame does not have any null values.')
        return handl_data


    def one_hot_encoder_one(self, feature, keep_first=True):
        """Apply one hot encoding for a categorical column."""
        try:
            oh = OneHotEncoder()
            encoding_data = self.reading_files()
            oh_df = pd.DataFrame(oh.fit_transform(encoding_data[[feature]]).toarray())
            oh_df.columns = oh.get_feature_names_out()

            for col in oh_df.columns:
                oh_df.rename({col: f'{feature}_' + col.split('_')[1]}, axis=1, inplace=True)

            new_data = pd.concat([encoding_data, oh_df], axis=1)
            new_data.drop(feature, axis=1, inplace=True)

            if not keep_first:
                new_data = new_data.iloc[:, 1:]

            return new_data
        except Exception as e:
            print(f"An error occurred: {e}")


    def target_encoding(self, column, target):
        """Apply target encoding for a categorical column."""
        try:
            grouped = self.reading_files()[[column, target]].groupby(column, as_index=False).mean()
            empty_dict = {}
            target_encoded_df = self.reading_files()
            for i in range(len(grouped)):
                empty_dict[grouped.iloc[i, 0]] = grouped.iloc[i, 1]
            target_encoded_df[column] = target_encoded_df[column].map(lambda x: empty_dict[x])
            return target_encoded_df
        except Exception as e:
            print(f"An error occurred: {e}")

    def label_encoder(self, feature):
        """Apply label encoding for a categorical column."""
        try:
            le = LabelEncoder()
            label_encoded_df = self.reading_files()
            label_encoded_df[feature] = le.fit_transform(label_encoded_df[feature])
            return label_encoded_df
        except Exception as e:
            print(f"An error occurred: {e}")
    def ordinal_encoder(self, feature, feature_rank):
        """Apply ordinal encoding for a categorical column."""
        try:
            ordinal_dict = {}
            ordinal_encoded_df = self.reading_files()
            for i, feature_value in enumerate(feature_rank):
                ordinal_dict[feature_value] = i + 1
            ordinal_encoded_df[feature] = ordinal_encoded_df[feature].map(lambda x: ordinal_dict[x])
            return ordinal_encoded_df
        except Exception as e:
            print(f"An error occurred: {e}")

# Example Usage
# df = DataProcessor("your_dataset_path")
# df.reading_files()
#   processor = DataProcessor("C:/Downloads/archive/diamonds.csv")
#   processor.reading_files()
# df.Data_Summary()
#   processor.Data_Summary()
# df.Handling_Missing_Values()
#   processor.Handling_Missing_Values()
# df.one_hot_encoder_one('your_column_name')
#   processor.one_hot_encoder_one('cut')
# df.target_encoding('your_column_name', 'target_column')
#   processor.target_encoding('cut','price')
# df.label_encoder('your_column_name')
#  processor.label_encoder('cut')
# df.ordinal_encoder('your_column_name','ordered list of unique values from your column')
#   processor.ordinal_encoder('cut',['Ideal','Premium','Good','Very Good','Fair'])
