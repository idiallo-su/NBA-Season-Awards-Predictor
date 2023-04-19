#Testing NER capabilities for future project use.
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

class Player:
    allAliases = []
    #making a user instance
    def __init__(self, name, aliases, stats, sentScore):
        self.name = name
        self.aliases = aliases
        self.stats = stats
        self.sentScores = sentScore



    #user methods----------------------------------------------------------------------------
    def checkMention(self, ls, text): #determine if a player is mentioned by name or alias
        if self.name in ls:
            return True
        #check aliases
        for alias in text:
            if alias in text:
                return True
        return False
    
    #def getStats(self, str): #retrieve stats from documents if not initally given

kj = Player("Lebron James", ["King James", "The Chosen One"], 0, 0)

doc = nlp("King James is on the Los Angeles Lakers.")

people = []
for ent in doc.ents:
    if (ent.label_ == "PERSON"):
         people.append(ent.text)
    print(ent.text + "|" + ent.label_)
    

print(kj.checkMention(people, doc))
