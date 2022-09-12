import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_basic_info(df):
    print("Basis data properties \n")
    print("\t **********\n")
    print("Data types: \n")
    print(df.dtypes)
    print("\n\t **********\n")
    print("Columns summary: \n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.describe(include="all"))


def import_data(path):
    df = pd.read_csv(path)
    # print_basic_info(df)

    # do not include children and young adults in statistics
    df.drop(df[df["smoking_status"] == "Unknown"].index, inplace=True)
    df.drop(df[df["work_type"] == "children"].index, inplace=True)  # just to be sure, but rather redundant
    df.drop(df[df["age"] <= 25].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # replace 'gender' column with dummy variables
    dummy_variable_1 = pd.get_dummies(df["gender"])
    df = pd.concat([df, dummy_variable_1], axis=1)
    df.drop("gender", axis=1, inplace=True)

    # replace 'work_type' column with dummy variables
    dummy_variable_2 = pd.get_dummies(df["work_type"])
    df = pd.concat([df, dummy_variable_2], axis=1)
    df.drop("work_type", axis=1, inplace=True)

    # replace 'Residence_type' column with dummy variables
    dummy_variable_3 = pd.get_dummies(df["Residence_type"])
    df = pd.concat([df, dummy_variable_3], axis=1)
    df.drop("Residence_type", axis=1, inplace=True)

    # replace 'smoking_status' column with dummy variables
    dummy_variable_4 = pd.get_dummies(df["smoking_status"])
    df = pd.concat([df, dummy_variable_4], axis=1)
    df.drop("smoking_status", axis=1, inplace=True)

    # change 'ever_married' column's type from object to int
    df["ever_married"].replace("Yes", "1", inplace=True)
    df["ever_married"].replace("No", "0", inplace=True)
    df["ever_married"] = df["ever_married"].astype("int")

    # create categories for glucose level
    bins1 = [min(df["avg_glucose_level"]), 140, 199, max(df["avg_glucose_level"])]
    group_names1 = ["Normal", "Pre-diabetes", "Diabetes"]
    df["glucose_level"] = pd.cut(df["avg_glucose_level"], bins1, labels=group_names1, include_lowest=True)

    f1 = plt.figure()
    plt.bar(group_names1, df["glucose_level"].value_counts().sort_index())
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=-45)
    plt.tight_layout()
    plt.show()

    # create categories for bmi
    bins2 = [min(df["bmi"]), 18.5, 25.0, 30.0, 35.0, 40.0, max(df["avg_glucose_level"])]
    group_names2 = ["Underweight", "Normal-weight", "Overweight",
                    "class-I-Obesity", "class-II-Obesity", "class-III-Obesity"]
    df["BMI_norms"] = pd.cut(df["bmi"], bins2, labels=group_names2, include_lowest=True)

    f2 = plt.figure()
    plt.bar(group_names2, df["BMI_norms"].value_counts().sort_index())
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=-45)
    plt.tight_layout()
    plt.show()

    print_basic_info(df)
