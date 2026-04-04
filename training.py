import spacy
from spacy.training.example import Example
import random 

nlp = spacy.blank("en")  # ! -> blank English model (no pretrained data)
ner = nlp.add_pipe('ner')  # ! -> adding NER pipeline to the model

TRAIN_DATA = [
    ("Rice crops has blast disease",
        {"entities":[(0,4,"CROP"),(15,20,"DISEASE")]}),  # ? -> dictionary hai (text + entity positions)

    ("Wheat show brown spot",
        {"entities":[(0,5,"CROP"),(11,21,"DISEASE")]}),

    ("Leaves have yellowing symptoms",
        {"entities":[(12,30,"SYMPTOM")]}),

    ("Apply carbendazim to control blast",
        {"entities":[(6,18,"CROP"),(29,35,"DISEASE")]}),

    ("Maize plant suffer from leaf blight",
        {"entities":[(0,5,"CROP"),(29,38,"DISEASE")]}),   
]

# ! -> add labels to NER before training (important step, warna model train nahi karega)
for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

optimizer = nlp.begin_training()  # ! -> initialize training optimizer

for epoch in range (30):
    random.shuffle(TRAIN_DATA)  # ! -> shuffle data for better learning each epoch
    losses = {}

    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)  # ! -> convert text to Doc object
        example = Example.from_dict(doc, annotations)  # ! -> create training example
        nlp.update([example], drop =0.3, losses=losses)  # ! -> update model (drop = regularization)

    print(f"Epoch {epoch} Loss: {losses}")  # ! -> track training loss

nlp.to_disk('ner_model')  # ! -> save trained model into folder 'ner_model'
print('Model was trained successfully!!!')