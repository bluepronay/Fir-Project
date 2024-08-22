import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Ensure necessary NLTK data is downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')


def preprocessing(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Define the set of stop words
    stop_words = set(stopwords.words('english'))
    
    # Remove stop words from the list of words
    words = [word for word in words if word not in stop_words]
    
    # Initialize the Porter stemmer
    stemmer = PorterStemmer()
    
    # Stem each word in the list of words
    words = [stemmer.stem(word) for word in words]
    
    # Join the list of words into a single string
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text


import pickle  #complex data structure ko store krna
with open('preprocess_data.pkl','wb') as  file :
  pickle.dump(new_ds,file)
new_ds=pickle.load(open('preprocess_data.pkl','rb'))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
model=SentenceTransformer('paraphrase-MiniLM-L6-v2')



def suggest_sections(complaint,dataset,min_suggestions=5):
    preprocessed_complaint=preprocessing(complaint)
    complaint_embedding=model.encode(preprocessed_complaint)
    section_embedding=model.encode(dataset['Combo'].tolist())
    similarities=util.pytorch_cos_sim(complaint_embedding,section_embedding)[0]
    similarity_threhold=0.2
    relevant_indices=[]
    while len(relevant_indices)<min_suggestions and similarity_threhold>0:
        relevant_indices=[i for i, sim in enumerate(similarities)if sim>similarity_threhold]
        similarity_threhold-=0.5 #st=st-0.5
        sorted_indices=sorted(relevant_indices,key=lambda i: similarities[i],reverse=True)
        suggestions=dataset.iloc[sorted_indices][['Description','Offense','Punishment','Cognizable','Bailable','Court','Combo']].to_dict(orient='records')
        return suggestions


# complaint=input("Enter crime description")
# suggest_sections=suggest_sections(complaint,new_ds)

from tkinter import Tk,Label,Entry,Text,Button,END


# if(suggest_sections):
#     print("Suggested Section are :")
#     for suggestion in suggest_sections:
#         print(f"Description :{suggestion['Description']}")
#         output_text.insert(END,f"Description :{suggestion['Description']}\n")
#         print(f"Offense :{suggestion['Offense']}")
#         output_text.insert(END,f"Offense :{suggestion['Offense']}\n")
#         print(f"Punishment: {suggestion['Punishment']}")
#         output_text.insert(END,f"Punishment: {suggestion['Punishment']}\n")
#         print("__________________________________________________________________________________________")

# else:
#     print("No record is found..")
#     output_text.insert(END,"No record is found..")

def get_suggestion():
    complaint=complaint_entry.get()
    suggestions=suggest_sections(complaint,new_ds)
    output_text.delete(1.0,END)
    if suggestions :
        output_text.insert(END,"Suggested IPC Sections are ")
root=Tk()
root.title("IPC Section Suggestion ")
complaint_label=Label(root,text="Enter crime description")  #banaya
complaint_label.pack()                                      #display
complaint_entry=Entry(root,width=100)
complaint_entry.pack()
suggest_button=Button(root,text="Get Suggestions",command=suggest_sections)   #iss function ko call krega
suggest_button.pack()
output_text=Text(root,width=100,height=20)
output_text.pack()
root.main()