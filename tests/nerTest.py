#Testing NER capabilities for future project use.
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("Lebron James is on the Los Angeles Lakers.")
people = []
for ent in doc.ents:
    if (ent.label_ == "PERSON"):
         people.append(ent.text)
    print(ent.text, "|", ent.label_ + "|") 

print("PLAYERS: ")
print(people)