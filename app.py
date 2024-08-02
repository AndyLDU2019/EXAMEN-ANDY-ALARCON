
from transformers import pipeline # huggin face
from flask_cors import CORS #restringuir las solicitudes http, manejar las solicitudes 
from flask import Flask, request, jsonify
from streamlit as st
import requests

# Configurar la URL del servidor Flask
flask_url = 'http://localhost:8008/procesar'
str.title('Ingrese su texto para clasificar')

# Entrada de texto del usuario
texto = str.text_area('Ingrese el texto para clasificar:', '')

if str.button('Clasificar'):
    if texto:
        # Enviar la solicitud POST al servidor Flask
        response = requests.post(flask_url, data={'texto': texto})

        if response.status_code == 200:
            # Obtener la etiqueta clasificada
            resultado = response.json()
            str.write('La etiqueta más probable es:', resultado['label'])
        else:
            str.write('Error en la solicitud al servidor Flask.')
    else:
        str.write('Por favor, ingrese el texto a clasificar.')



app= Flask(__name__)
CORS(app)

@app.route( '/procesar', methods=['POST'])

def procesar_datos():
    #texto = request.form.get('texto','El lanzamiento del nuevo iPhone ha causado un gran revuelo en la industria de la tecnología. Las acciones de Apple subieron un 5% después del anuncio.')
    texto = request.form.get('texto',"")
    candidate_labels = ['deportes','política', 'religión','cine']

    
    classifier = pipeline("zero-shot-classification")
    model="Facebook/basrt-large-mnli" #es un modelo de lenguaje preentrenado que está diseñado para manejar múltiples idiomas.

    resultado = classifier(texto, candidate_labels)
    max_score = resultado ['scores'].index(max(resultado['scores']))
    max_label = resultado ['labels'] [0]

    return jsonify ({'label': max_label})

#validamos


if __name__ == '__main__':
    app.run(port=8008)









