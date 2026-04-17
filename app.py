from fastapi import FastAPI
from pydantic import BaseModel  # ! -> input data define aur validate karne ke liye use hota hai
import spacy

app = FastAPI()

nlp = spacy.load("ner_model")  # ! -> trained NER model load kar rahe hain

class InputText(BaseModel):
    text: str  # ! -> input field (jo text analyze karna hai)

# ! -> POST API endpoint create kar rahe hain

@app.post("/predict")
def predict(data: InputText):
    doc = nlp(data.text)  # ! -> input text ko trained model se process kar rahe hain

    return {
        "entities": [
            {"text": ent.text, "label": ent.label_}  # ! -> entity text aur uska label return kar rahe hain
            for ent in doc.ents
        ]
    }