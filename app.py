from fastapi import FastAPI
from pydantic import BaseModel  # ! -> used to define and validate input data
import spacy

app = FastAPI()

nlp = spacy.load("ner_model")  # ! -> load your trained NER model

class InputText(BaseModel):
    text: str  # ! -> input field (text to analyze)

# ! -> creating POST API endpoint

@app.post("/predict")
def predict(data: InputText):
    doc = nlp(data.text)  # ! -> process input text using trained model

    return {
        "entities": [
            {"text": ent.text, "label": ent.label_}  # ! -> return entity text + label
            for ent in doc.ents
        ]
    }