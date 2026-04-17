import spacy
from spacy.training.example import Example
import random 

nlp = spacy.blank("en")  # ! -> blank English model (no pre-trained data)
ner = nlp.add_pipe('ner')  # ! -> NER pipeline model me add kar rahe hain

TRAIN_DATA = [
    ("Rice crops has blast disease",
        {"entities":[(0,4,"CROP"),(16,29,"DISEASE")]}),  # ? -> yeh dictionary hai (text + entity positions)

    ("Wheat show brown spot",
        {"entities":[(0,5,"CROP"),(11,21,"DISEASE")]}),

    ("Leaves have yellowing symptoms",
        {"entities":[(12,30,"SYMPTOM")]}),

    ("Apply carbendazim to control blast disease",
        {"entities":[(6,18,"CHEMICAL"),(30,43,"DISEASE")]}),

    ("Maize plant suffer from leaf blight",
        {"entities":[(0,5,"CROP"),(29,41,"DISEASE")]}),   
]

# ! -> training se pehle NER ko labels dena zaroori hai, warna model train nahi karega
for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

optimizer = nlp.begin_training()  # ! -> training ke liye optimizer initialize kar rahe hain

for epoch in range (30):
    random.shuffle(TRAIN_DATA)  # ! -> har epoch me data shuffle karte hain taaki model better learn kare
    losses = {}

    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)  # ! -> text ko Doc object me convert kar rahe hain
        example = Example.from_dict(doc, annotations)  # ! -> training example create kar rahe hain
        nlp.update([example], drop =0.3, losses=losses)  # ! -> model update kar rahe hain (drop = regularization)

    print(f"Epoch {epoch} Loss: {losses}")  # ! -> training loss track kar rahe hain

nlp.to_disk('ner_model')  # ! -> trained model ko 'ner_model' folder me save kar rahe hain
print('Model was trained successfully!!!')