import os
import spacy 
import numpy as np
from transformers import pipeline
#__________________________________________________
# Web links Handler
import requests
# Backend
import torch
# Image Processing
from PIL import Image
# Transformer and pre-trained Model
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
# Managing loading processing
from tqdm import tqdm
# Assign available GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

#________________________________________________________________________________
#________________________________________________________________________________
#_____This code is for image captionig___________________________________________
#________________________________________________________________________________
#________________________________________________________________________________


# Loading a fine-tuned image captioning Transformer Model

# ViT Encoder-Decoder Model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
# Corresponding ViT Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# Image processor
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Accesssing images from the web
import urllib.parse as parse
import os
# Verify url
def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# Load an image
def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    

# Image inference
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # Preprocessing the Image
    img = image_processor(image, return_tensors="pt").to(device)
    # Generating captions
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

#________________________________________________________________________________
#________________________________________________________________________________
#___________________THIS CODE IS FOR TEXT CLASSIFICATION_________________________
#________________________________________________________________________________
#________________________________________________________________________________


class image_pos:
    def __init__(self, link, adj, noun, verb):
        self.link = link
        self.adj = adj
        self.noun = noun
        self.verb = verb

# Load a pre-trained BERT model for sentiment analysis
sentiment_analysis = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_sm")

 # load en_core_web_sm of English for vocabluary, syntax & entities
def image_classification2(Text):
    
    #nlp = en_core_web_sm.load()
    docs = nlp(Text)
    POS_needed = ["ADJ", "NOUN", "VERB"]
    POS_of_sentence = np.array([[],[]])
    adj = np.array([])
    noun = np.array([])
    verb = np.array([])
   
    for word in docs:
        if word.pos_ == "ADJ":
            adj = np.append(adj,word.text)
            
        elif word.pos_ == "NOUN":
            noun = np.append(noun,word.text)
        
        elif word.pos_ == "VERB":
            verb = np.append(verb,word.text)
            
    link = ""        
    final_image = image_pos(link, adj, noun, verb)                  
    return(final_image)



#________________________________________________________________________________
#_______________________________HERE WE HAVE THE MAIN____________________________

folder_path = "/media/mitos/E054078554075DA2/Github PROYECTS/Image Captioning with Chatgpt2/imagenesprueba"


for filename in os.listdir(folder_path):

    filepath = os.path.join(folder_path, filename)
    if os.path.isfile(filepath):
        
        CAPTION = get_caption(model, image_processor, tokenizer, filepath)
        print("File name: ", filename)
        print("PREDICT CAPTION : %s" %(CAPTION))
        text = CAPTION
        prueba = image_classification2(text)
        prueba.link = filepath
        print("URL/PATH: {}".format(prueba.link))
        print("Adjetivos: {}".format(prueba.adj))
        print("Objetos: {}".format(prueba.noun))
        print("Verbos: {}".format(prueba.verb))
        #__________Here we make the sentiment analysis_____________
        doc = nlp(CAPTION)
        text_tokens = [token.text for token in doc]
        preprocessed_sentence = " ".join(text_tokens)
        result = sentiment_analysis(preprocessed_sentence)
        print("SENTIMENT CLASSIFICATION : %s" %(result))
        print()
        print()

