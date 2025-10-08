import gradio as gr
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
try:
    xgb_model = joblib.load('xgboost_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = LabelEncoder()
    # In a real deployment, you would save and load the fitted label encoder
    # For demonstration, fit with all possible classes
    label_encoder.fit(['P1', 'P2', 'P3', 'P4'])
    # Load the description file for feature labels
    description_df = pd.read_excel("Description.xlsx")
    feature_description_mapping = description_df.set_index('Variable Name')['Description'].to_dict()
except FileNotFoundError:
    print("Error: Model, scaler, or description file not found. Please ensure 'xgboost_model.joblib', 'scaler.joblib', and 'Description.xlsx' are in the same directory.")
    exit() # Exit if files are missing

# Assuming sorted_importances is available from your notebook analysis
# In a real script, you would either recalculate this or save/load the top features list
# For demonstration, using the top 10 features identified in the notebook
# You would typically load or define top_10_features based on your analysis results
# For now, hardcoding based on previous analysis:
top_10_features = ['enq_L3m', 'Age_Oldest_TL', 'pct_PL_enq_L6m_of_ever', 'max_recent_level_of_deliq', 'num_std_12mts', 'time_since_recent_enq', 'recent_level_of_deliq', 'max_deliq_12mts', 'num_deliq_6_12mts', 'Age_Newest_TL']


def predict_approved_flag_gradio(*args):
    """
    Preprocesses raw input data and predicts the Approved_Flag using a saved XGBoost model.

    Args:
        *args: Input features from the Gradio interface in the order defined by the inputs list.

    Returns:
        str: The predicted Approved_Flag ('P1', 'P2', 'P3', or 'P4').
             Returns an error message if preprocessing or prediction fails.
    """
    # Ensure the number of inputs matches the expected number of features
    if len(args) != len(top_10_features):
        return "Error: Incorrect number of input features provided."

    raw_data = dict(zip(top_10_features, args))

    try:
        # Convert raw_data to a pandas DataFrame
        input_df = pd.DataFrame([raw_data])

        # Define columns to be removed (based on previous analysis)
        # These columns were removed due to having a high percentage of -99999 values
        columns_to_be_removed = ['time_since_first_deliquency', 'time_since_recent_delinquency',
                                 'max_delinquency_level', 'max_deliq_6mts', 'CC_utilization',
                                 'PL_utilization', 'max_unsec_exposure_inPct']

        # Handle missing values (-99999) for the *original* full set of columns
        # This is crucial because the model was trained on the full set.
        # For features NOT in top_10_features, we need to ensure they are handled correctly.
        # A robust solution would involve saving the imputation logic or values from training.
        # For this simplified interface, we will fill in the values for the full feature set
        # and then perform preprocessing as done during training.

        # Define the full list of columns the model was trained on (from df_encoded)
        # In a real deployment, save and load this list.
        training_columns = ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'max_deliq_12mts', 'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 'num_lss_12mts', 'recent_level_of_deliq', 'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag', 'EDUCATION', 'MARITALSTATUS_Single', 'GENDER_M', 'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan', 'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others', 'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan', 'first_prod_enq2_HL', 'first_prod_enq2_PL', 'first_prod_enq2_others']

        # Create a DataFrame with all training columns and fill with default values (e.g., 0)
        # A more robust solution would use training data means or median for imputation
        full_feature_df = pd.DataFrame(0.0, index=[0], columns=training_columns)

        # Populate the full DataFrame with the provided top 10 feature values
        for feature, value in raw_data.items():
             if feature in full_feature_df.columns:
                  # Handle ordinal encoding for EDUCATION if it's in top_10
                  if feature == 'EDUCATION':
                       education_mapping = {'OTHERS': 1, 'SSC': 1, '12TH': 2, 'UNDER GRADUATE': 3, 'GRADUATE': 3, 'PROFESSIONAL': 3, 'POST-GRADUATE': 4}
                       full_feature_df[feature] = education_mapping.get(value, 1) # Use 1 as default for unknown
                  # Handle one-hot encoding for other categorical features if they are in top_10
                  elif feature in ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
                       # Assuming one-hot encoded column names are like 'CATEGORY_VALUE'
                       # This requires knowing the possible values from training
                       if feature == 'MARITALSTATUS' and value == 'Single':
                           full_feature_df['MARITALSTATUS_Single'] = 1.0
                       elif feature == 'GENDER' and value == 'M':
                           full_feature_df['GENDER_M'] = 1.0
                       elif feature in ['last_prod_enq2', 'first_prod_enq2']:
                            encoded_col_name = f'{feature}_{value}'
                            if encoded_col_name in full_feature_df.columns:
                                full_feature_df[encoded_col_name] = 1.0
                  else:
                       # For numerical features in top_10
                       full_feature_df[feature] = value

        # --- Apply the same preprocessing steps as during training ---
        # Note: This assumes the preprocessing steps are simple (imputation and scaling).
        # Complex steps might require saving and loading the preprocessor object.

        # Handle remaining -99999 values in numerical columns (those not in top 10 but in training_columns)
        # A robust solution would load the mean/median values from training data for imputation
        for col in full_feature_df.columns:
             if full_feature_df[col].dtype != 'object':
                  # Replace -99999 with the mean of non -99999 values (assuming some data might still have it)
                  mean_val = full_feature_df[col][full_feature_df[col] != -99999].mean() if not full_feature_df[col][full_feature_df[col] != -99999].empty else 0
                  full_feature_df[col] = full_feature_df[col].replace(-99999, mean_val)


        # Apply scaling to numerical features using the loaded scaler
        numerical_cols = full_feature_df.select_dtypes(include=np.number).columns.tolist()
        full_feature_df[numerical_cols] = scaler.transform(full_feature_df[numerical_cols])


        # Make prediction
        prediction_encoded = xgb_model.predict(full_feature_df)

        # Decode the prediction
        predicted_flag = label_encoder.inverse_transform(prediction_encoded)

        return predicted_flag[0]

    except Exception as e:
        return f"Error during prediction: {e}"


# Define the input components for Gradio based on the top 10 features with user-friendly labels
input_components = []
for feature in top_10_features:
    # Get the user-friendly label from the description file, or use the technical name if not found
    feature_label = feature_description_mapping.get(feature, feature)

    # Define appropriate Gradio input types for the top 10 features
    # This mapping needs to be accurate based on the actual data types and ranges
    if feature == 'enq_L3m':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'Age_Oldest_TL':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'pct_PL_enq_L6m_of_ever':
        input_components.append(gr.Number(label=feature_label, minimum=0.0, maximum=1.0))
    elif feature == 'max_recent_level_of_deliq':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'num_std_12mts':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'time_since_recent_enq':
        input_components.append(gr.Number(label=feature_label, minimum=-99999, precision=0)) # Keep -99999 handling
    elif feature == 'recent_level_of_deliq':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'max_deliq_12mts':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'num_deliq_6_12mts':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    elif feature == 'Age_Newest_TL':
        input_components.append(gr.Number(label=feature_label, minimum=0, precision=0))
    # Add more specific input types if other categorical features happen to be in the top 10
    # For example:
    # elif feature == 'EDUCATION':
    #      input_components.append(gr.Dropdown(['GRADUATE', 'SSC', '12TH', 'UNDER GRADUATE', 'POST-GRADUATE', 'OTHERS', 'PROFESSIONAL'], label=feature_label))


# Create the Gradio interface
iface = gr.Interface(
    fn=predict_approved_flag_gradio,
    inputs=input_components,
    outputs="text",
    title="Approved Flag Prediction App (Top 10 Features)",
    description="Enter the details for the top 10 features to predict the Approved Flag."
)

# To run the Gradio app, you would use:
# iface.launch()
# In Colab, you can use inline=True to display it within the notebook:
# iface.launch(inline=True)

if __name__ == "__main__":
    print("Gradio interface created. You can launch it using `iface.launch()`")
    # Use share=True when running in environments like Colab or Kaggle
    iface.launch(share=True)
