import os
# import pandas as pd
# import numpy as np
from string import punctuation
import glob
# import nltk
import re
import unidecode
import math
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#When running for the first time, please run the following two lines of code:
# nltk.download('stopwords')
# nltk.download('punkt')

punctuation = list(punctuation)
stop_words = stopwords.words('english')

def processing(text):
    new_text = text.strip().lower()
    new_text = new_text.replace('  ', '')
    new_text = unidecode.unidecode(new_text)
    new_text = new_text.replace("'", '')
    new_text = new_text.replace("-", '')
    new_text = re.sub("[^a-zA-Z\s]+", '', new_text)
    tokens = word_tokenize(new_text)
    #Save tokens that are not stop words and aren't punctuation marks
    
    cleaned_tokens = [token for token in tokens if token not in stop_words 
                      and token not in punctuation and len(token) > 1]
    ps = PorterStemmer()
    stem_tokens = [ps.stem(token) for token in cleaned_tokens]

    return stem_tokens

def read_processed_file(file):
    # Función que lee un archivo procesado
    f = open(file, 'r')
    text = f.read()
    f.close()
    new_text = text.strip()
    tokens = new_text.split(' ')
    return tokens

def create_processed_file(directory, name_file, tokens):
    #Create the processed text (now tokens) into another file
    with open(os.path.join(directory, name_file), 'w') as file:
        file.write(" ".join(tokens))

if not os.path.exists('procesados'):
    os.mkdir('procesados')

#Define directory of un-processed files
directory = 'corpus'

#Number of files and a sorted list of them
ordered_files = sorted(glob.glob(f'{directory}/*'))
num = len(ordered_files)

#Iterate over the files in thar directory
tf = {}
n = {}
for id, filename in enumerate(ordered_files):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    tokens = processing(text)
    
    create_processed_file("procesados", filename.split('/')[-1], tokens)

    for t in set(tokens):
        if t not in tf:
            tf[t] = {}
        tf[t][id] = 1 + math.log2(tokens.count(t))
        # guarda el conteo de las palabras
        n[t] = n.get(t,0) + tokens.count(t)

# Obtain idf=log2(N/ni) where
#           N : Total number of docs
#           ni :Number of docs where N appears

idf = {k:math.log2(num/v) for (k,v) in n.items()}

tf_idf = {}
for k,v in idf.items():
    diccionario = tf.get(k)
    tf_idf[k] = {k1: v1 * v for k1, v1 in diccionario.items()}

directory = 'procesados'

ordered_files = sorted(glob.glob(f'{directory}/*'))
num = len(ordered_files)

longitud = []

for id, filename in enumerate(ordered_files):
    valor = 0
    
    tokens = read_processed_file(filename)

    for t in set(tokens):
        try:
            valor += tf_idf[t][id]
        except KeyError:
            pass
        
    longitud.append(math.sqrt(math.pow(valor,2)))

def save_file(data, name):
    with open(f"{name}.pkl", 'wb') as fp:
        pickle.dump(data, fp)

def read_file(name):
    with open(f"{name}.pkl", 'rb') as fp:
        return pickle.load(fp)

# save dictionary to tf_idf, idf and longitud file
save_file(tf_idf, "tf_idf")
save_file(idf, "idf")
save_file(longitud, "longitud")

#Queries with only 1 word:
def consulta_one_word():
    consulta = input("Please make the query (1 word only): ")
    num_results=int(input("How many results would you like to see? "))
    consulta_procesada= processing(consulta) #El mismo pre procesamiento que se le hizo a los documentos se le aplica a la consulta
    palabra = consulta_procesada[0]
    
    try:
        resultados_ranking = {i: tf_idf[palabra][i] / longitud[i] for i in tf_idf[palabra].keys()} #Las llaves son los documentos
        ranking_ordenado=sorted(resultados_ranking.items(), key=lambda item: item[1],reverse=True)[:num_results] #Aquí ya los ordena
        for key,value in ranking_ordenado:
            print(f"Document number {key} with a weight of {value}")
    except KeyError:
        print("No results")

#Queries with more than 1 word and an AND clause
#Example:"how many galaxies"
def consulta_and():
    consulta_and = input("Please make the query. This is an AND query.\n")
    num_results_and=int(input("How many results would you like to see? "))
    terminos = processing(consulta_and)  # Apply the same preprocessing as done for the documents to the query
    #Find documents that have the query terms (already processed)
    try:
        docs_comunes = set.intersection(*(set(tf[i].keys()) for i in terminos))
        resultados_and={}
        for doc in docs_comunes:
            total_tfidf = sum(tf_idf[termino][doc] for termino in terminos) #Se suman los tf_idf de cada termino a nivel documento
            resultados_and[doc]=total_tfidf

        resultados_and_ordenados=sorted(resultados_and.items(),key=lambda item: item[1],reverse=True)[:num_results_and]
        for key, value in enumerate(resultados_and_ordenados):
            print(f"Document {ordered_files[value[0]].split('/')[-1]} with a weight of {value[1]}")

    except KeyError:
        print("No results")

#Specific phrase query
#Example: black hole
def consulta_frase():
    consulta_comillas=input("Please make your query (between quotes " "): ").strip('"').lower()
    num_results_comillas=int(input("How many results would you like to see? "))
    resultados_comillas={}
    try:
        for filename in ordered_files:
            file = open(filename, 'r')
            text = file.read().lower()
            file.close()
            if consulta_comillas in text:
                filename = filename.split('/')[-1]
                resultados_comillas[filename]=text.count(consulta_comillas) #Elegimos hacer el ranking únicamente contando cuántas veces aparece la palabra

        resultados_comillas_ordenados=sorted(resultados_comillas.items(),key=lambda item: item[1],reverse=True)[:num_results_comillas]
    except KeyError:
        print("No results")

    for doc, num in resultados_comillas_ordenados:
        print(f"Document {doc} with a number of repetitions of {num}")

def menu():
    print("Information recovery program")
    print("""Please type the number of the type of query you would like to make.
          1.- One word
          2.- AND
          3.- Phrase\n""")
    choice = int(input("Type the number: "))
    if choice == 1:
        consulta_one_word()
    elif choice ==2:
        consulta_and()
    elif choice==3:
        consulta_frase()

menu()


