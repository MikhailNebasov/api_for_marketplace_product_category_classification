import json
import dill
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Form(BaseModel):
    product_id: int
    sale: str
    shop_id: int
    shop_title: str
    rating: float
    text_fields: str


class Prediction(BaseModel):
    prediction: int


app = FastAPI()

file_name = 'nlp_model.pkl'
with open(file_name, 'rb') as file:
    model = dill.load(file)


class Nlp_API():

    @app.get('/status')
    def status():
        return "I'm OK"


    @app.get('/version')
    def version():
        return model['metadata']


    def prepare_data(dfr):
        dataset = dfr['text_fields']
        shop_list = dfr['shop_title']
    
        data_for_tokenization = json.loads(dataset)['title'] + ' ' + json.loads(dataset)['title'] + ' ' + json.loads(dataset)['title'] + ' ' + json.loads(dataset)['title'] + ' ' + shop_list + ' ' + shop_list + ' ' + json.loads(dataset)['description'] + ' ' + str(json.loads(dataset)['attributes']) + ' ' + str(json.loads(dataset)['custom_characteristics']) + ' ' + str(json.loads(dataset)['defined_characteristics'])

        # Удаление html тегов
        flag = True
        while flag:
            pos1 = data_for_tokenization.find('<')
            if pos1 == -1:
                flag = False
            pos2 = data_for_tokenization.find('>')
            if pos2 == -1:
                flag = False
            if pos1 != -1:
                if pos2 != -1:
                    if pos1 < pos2:
                        data_for_tokenization = data_for_tokenization[0 : pos1] + ' ' + data_for_tokenization[pos2 + 1:]
                    else:
                        flag = False

        # Удаление зарезервированных символов html
        data_for_tokenization = data_for_tokenization.replace('&nbsp;',' ')
        data_for_tokenization = data_for_tokenization.replace('&lt;',' ')
        data_for_tokenization = data_for_tokenization.replace('&gt;',' ')
        data_for_tokenization.strip()

        # Удаление возможных двойных пробелов
        flag = True
        while flag:
            pos1 = data_for_tokenization.find('  ')
            if pos1 == -1:
                flag = False
            else:
                data_for_tokenization = data_for_tokenization.replace('  ',' ')
        return [data_for_tokenization]


    @app.post('/predict', response_model=Prediction)
    def predict(form: Form):
        return {'prediction': model['model'].predict(model['stage2'].transform(model['stage1'].transform(Nlp_API.prepare_data(form.model_dump()))))}


uvicorn.run(app, host="127.0.0.1", port=8000)