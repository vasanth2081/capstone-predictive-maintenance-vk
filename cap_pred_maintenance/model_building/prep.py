# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/v-vasanth2009/capstone-pred-main-vk-12052026/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unnammed : 0 column in dataset
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

def feature_engineering(df):

    df = df.copy()

    df['temp_diff'] = (
        df['Lub oil temp'] -
        df['Coolant temp']
    )

    df['engine_stress'] = (
        df['Engine rpm'] *
        df['Coolant temp']
    )

    df['heat_index'] = (
        df['Lub oil temp'] +
        df['Coolant temp']
    ) / 2

    df['pressure_ratio'] = (
        df['Fuel pressure'] /
        (df['Lub oil pressure'] + 1e-5)
    )

    return df

df = feature_engineering(df)

target_col = 'Engine Condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Outlier Handling 
numerical_cols = X.select_dtypes(
    include=['int64', 'float64']
).columns

for col in numerical_cols:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    X[col] = X[col].clip(lower, upper)
    
# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle Missing Values
for col in Xtrain.columns:
    if Xtrain[col].dtype == 'object':
        fill_value = Xtrain[col].mode()[0]
    else:
        fill_value = Xtrain[col].median()

    Xtrain[col] = Xtrain[col].fillna(fill_value)
    Xtest[col] = Xtest[col].fillna(fill_value)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="v-vasanth2009/capstone-pred-main-vk-12052026",
        repo_type="dataset",
    )
