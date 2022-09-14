import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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

    # do not include people under age 30
    df.drop(df[df["smoking_status"] == "Unknown"].index, inplace=True)
    df.drop(df[df["work_type"] == "children"].index, inplace=True)  # just to be sure, but rather redundant
    df.drop(df[df["age"] < 30].index, inplace=True)
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
    bins1 = [min(df["avg_glucose_level"]), 140, max(df["avg_glucose_level"])]
    group_names1 = ["Normal", "High"]
    df["glucose_level"] = pd.cut(df["avg_glucose_level"], bins1, labels=group_names1, include_lowest=True)

    # f1 = plt.figure()
    # plt.bar(group_names1, df["glucose_level"].value_counts().sort_index())
    # ax = plt.gca()
    # ax.tick_params(axis='x', labelrotation=-45)
    # plt.tight_layout()
    # plt.show()

    # create categories for bmi
    bins2 = [min(df["bmi"]), 18.5, 25.0, 30.0, 35.0, 40.0, max(df["avg_glucose_level"])]
    group_names2 = ["Underweight", "Normal-weight", "Overweight",
                    "class-I-Obesity", "class-II-Obesity", "class-III-Obesity"]
    df["BMI_norms"] = pd.cut(df["bmi"], bins2, labels=group_names2, include_lowest=True)

    # f2 = plt.figure()
    # plt.bar(group_names2, df["BMI_norms"].value_counts().sort_index())
    # ax = plt.gca()
    # ax.tick_params(axis='x', labelrotation=-45)
    # plt.tight_layout()
    # plt.show()

    # create age categories
    bins3 = [min(df['age']), 40, 50, 60, 70, 80, max(df['age'])]
    group_names3 = ["30+", "40+", "50+", "60+", "70+", "80+"]
    df["age_groups"] = pd.cut(df['age'], bins3, labels=group_names3, include_lowest=True)

    f3 = plt.figure()
    plt.bar(group_names3, df["age_groups"].value_counts().sort_index())
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=-45)
    plt.tight_layout()
    plt.show()

    print_basic_info(df)
    path = "/home/karolina/PycharmProjects/brain_stroke_analysis/after_wrangling.csv"
    df.to_csv(path)
    return path


def get_indicators(path):
    df = pd.read_csv(path)
    alpha = 0.10
    mean = df['stroke'].mean() * 100
    print(f"\n\tPeople that had stoke: {mean}%\n ")

    df_group1 = df[['age_groups', 'stroke']]
    df_group1 = df_group1.groupby(['age_groups'], as_index=True).mean() * 100
    df_group1.rename(columns={'stroke': 'stroke %'}, inplace=True)
    print(df_group1)

    c, p = stats.chisquare(df_group1)
    print(f"P value for age is : {p}")
    if p <= alpha:
        print("Age is an indicator")
    else:
        print("Age is not an indicator")

    f4 = plt.figure()
    plt.plot(df_group1)
    plt.title("Percentage of people who have had a stroke depending on the age group")
    plt.xlabel("Age group")
    plt.ylabel("%")
    plt.show()

    df_group2 = df[['glucose_level', 'stroke']]
    df_group2 = df_group2.groupby(['glucose_level'], as_index=True).mean() * 100
    df_group2.rename(columns={'stroke': 'stroke %'}, inplace=True)
    print(df_group2)

    c, p = stats.chisquare(df_group2)
    print(f"P value for glucose level is : {p}")
    if p <= alpha:
        print("Glucose level is an indicator")
    else:
        print("Glucose level is not an indicator")

    f5 = plt.figure()
    xlabels = ["High", "Normal"]
    plt.bar(xlabels, df_group2['stroke %'])
    plt.title("Percentage of people who have had a stroke depending on the glucose level")
    plt.xlabel("Glucose level")
    plt.ylabel("%")
    plt.show()

    df_group3 = df[['BMI_norms', 'stroke']]
    df_group3 = df_group3.groupby(['BMI_norms'], as_index=True).mean() * 100
    df_group3.rename(columns={'stroke': 'stroke %'}, inplace=True)
    print(df_group3)

    c, p = stats.chisquare(df_group3)
    print(f"P value for BMI is : {p}")
    if p <= alpha:
        print("BMI is an indicator")
    else:
        print("BMI is not an indicator")

    f6 = plt.figure()
    xlabels = ["Underweight", "Normal-weight", "Overweight",
               "class-I-Obesity", "class-II-Obesity", "class-III-Obesity"]
    plt.bar(xlabels, df_group3['stroke %'])
    plt.title("Percentage of people who have had a stroke depending on the BMI")
    plt.xlabel("BMI norms")
    plt.ylabel("%")
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=-45)
    plt.tight_layout()
    plt.show()

