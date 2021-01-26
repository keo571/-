# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


class DataPreprocessing:

    def __init__(self, x_full_path, x_test_path, target_col_name):
        import pandas as pd
        self.x_train = pd.read_csv(x_full_path)
        self.x_train.dropna(axis=0, subset=[target_col_name], inplace=True)
        self.y_train = self.x_train[target_col_name]
        self.x_test = pd.read_csv(x_test_path)
        self.x_train.drop([self.y_train.name], axis=1, inplace=True)

    def show_data(self):
        # print("x_train:")
        # print(self.x_train.head())
        # print("y_train:")
        # print(self.y_train.head())
        # print("x_test:")
        # print(self.x_test.head())
        return self.x_train

    def pipeline(self, numerical_cols, categorical_cols):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import numpy as np

        # Preprocessing for numerical data
        def prep_num_data(num_cols, data, strategy):
            numerical_transformer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            if self.x_train.equals(data):
                out_data = pd.DataFrame(numerical_transformer.fit_transform(data[num_cols]))
                out_data.columns = data[num_cols].columns
            else:
                numerical_transformer.fit_transform(data[num_cols])
                out_data = pd.DataFrame(numerical_transformer.transform(data[num_cols]))
                out_data.columns = data[num_cols].columns
            return out_data

        # Preprocessing for categorical data
        def prep_cat_data(cat_cols, data):
            cat_onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
            imputed_cat_x_train = prep_num_data(cat_cols, self.x_train, "most_frequent")
            imputed_cat_x_test = prep_num_data(cat_cols, self.x_test, "most_frequent")

            if self.x_train.equals(data):
                out_data = pd.DataFrame(cat_onehot.fit_transform(imputed_cat_x_train))
                out_data.index = imputed_cat_x_train.index
            else:
                cat_onehot.fit_transform(imputed_cat_x_train)
                out_data = pd.DataFrame(cat_onehot.transform(imputed_cat_x_test))
                out_data.index = imputed_cat_x_test.index
            return out_data

        num_x_train = prep_num_data(num_cols=numerical_cols, data=self.x_train, strategy="median")
        num_x_test = prep_num_data(num_cols=numerical_cols, data=self.x_test, strategy="median")
        cat_x_train = prep_cat_data(cat_cols=categorical_cols, data=self.x_train)
        cat_x_test = prep_cat_data(cat_cols=categorical_cols, data=self.x_test)

        final_x_train = pd.concat([num_x_train, cat_x_train], axis=1)
        final_x_train.to_csv("final_x_train.csv", index=False)
        final_x_test = pd.concat([num_x_test, cat_x_test], axis=1)
        final_x_test.to_csv("final_x_test.csv", index=False)
        self.y_train.to_csv("y_train.csv", header=False, index=False)
        final_x_train_with_y = pd.concat([final_x_train, self.y_train], axis=1)
        final_x_train_with_y.to_csv("final_x_train_with_y.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_f_path = "/Users/qiyaowu/PycharmProjects/DataPreprocessingPipeline/train.csv"
    x_t_path = "/Users/qiyaowu/PycharmProjects/DataPreprocessingPipeline/test.csv"
    obj = DataPreprocessing(x_f_path, x_t_path, "Survived")
    # x_train = obj.show_data()
    # print(x_train.isnull().sum())
    numerical_col = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']
    categorical_col = ['Sex']
    obj.pipeline(numerical_cols=numerical_col, categorical_cols=categorical_col)
