# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml_model import preprocess_data, train_models
from sklearn.model_selection import train_test_split  

class Input(BaseModel):
    news: str

app = FastAPI()

data_fake = pd.read_csv(r'Fake.csv')
data_true = pd.read_csv(r'True.csv')


data = preprocess_data(data_fake, data_true)


x = data['text']
y = data['class']
x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25)


LR, DT, GB, RF, vectorization = train_models(x_train, y_train)

@app.put("/predict")
def predict_news(item: Input):
    testing_news = {"text": [item.news]}
    new_def_test = pd.DataFrame(testing_news)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return {
        "LR": output_label(pred_LR[0]),
        "DT": output_label(pred_DT[0]),
        "GB": output_label(pred_GB[0]),
        "RF": output_label(pred_RF[0])
    }

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not a Fake News"