import gradio as gr
import joblib
import pandas as pd

inputs = [gr.Dataframe(row_count = (3, "dynamic"), col_count=(13,"dynamic"), label="Input Data", interactive=1)]

outputs = [gr.Dataframe(row_count = (3, "dynamic"), col_count=(1, "fixed"), label="Predictions", headers=["Heart attack risk"])]
inference_model = joblib.load("tree_classification_of_heart_attacks.pkl")

df = pd.read_csv("Gui.csv")

def infer(input_dataframe):
  return pd.DataFrame(inference_model.predict(input_dataframe))

gr.Interface(fn = infer, inputs = inputs, outputs = outputs, examples = [[df.head(3)]]).launch()