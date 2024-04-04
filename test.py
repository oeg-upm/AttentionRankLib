
from transformers import BertTokenizer, TFBertModel, AutoModel, AutoTokenizer, RobertaTokenizer,RobertaModel

from src.attentionrank.preprocessing import preprocessing_module
from src.attentionrank.eval import evaluate_results


'''
#FacebookAI/roberta-base
modelname= 'roberta-base' #'roberta-base'  #'bert-base-uncased'

bertemb= ModelEmbedding(modelname,'roberta')
print(bertemb(['increased use']))

tokenizer = AutoTokenizer.from_pretrained(modelname)
print(tokenizer.tokenize(' increased use'))


'''
import spacy

# Cargar el modelo de Spacy para español
nlp = spacy.load("es_core_news_sm")

# Texto de ejemplo
texto = "Oxidación in situ, los experimentos se llevaron a cabo utilizando discos de 3 mm de diámetro con una superficie rectificada y pulida hasta un acabado de diamante de 1 μm. Luego, los discos de 3 mm se oxidaron en un Philips XL-30 FEG ESEM con un accesorio de platina caliente. La atmósfera oxidante utilizada fue aire de laboratorio a una presión de 266 Pa. Durante el experimento, se observó y se tomaron imágenes de la muestra utilizando una energía de haz primario de 20 kV y un detector de electrones secundario Everhart-Thornley. La muestra se calentó a una velocidad de 100°C/min hasta una temperatura de 700°C y se mantuvo a esta temperatura durante 8 minutos para estabilizar la platina y el microscopio. Luego la muestra fue calentada hasta una temperatura final de 900°C a la misma velocidad de calentamiento. El tiempo total de exposición de la muestra fue de 120 minutos antes de enfriarla a temperatura ambiente apagando las bobinas calefactoras. Luego, las muestras se examinaron en el LEO 1530VP FEGSEM con información química recopilada mediante EDS. Se produjeron secciones transversales y muestras de microscopio electrónico de transmisión (TEM) usando un FEI Nova Nanolab 600 de doble haz para fresado con haz de iones enfocados (FIB) perpendicular a los límites de fase para determinar su influencia en el desarrollo de óxido y se tomaron imágenes usando un Jeol 2000FX W- filamento TEM. Los mapas EDS de las muestras TEM se recolectaron utilizando el Nanolab 600 con un detector Scanning TEM (STEM) y un sistema EDAX Genesis EDS a un voltaje de aceleración de 30 kV."
texto='El mal comportamiento de oxidación es la principal barrera para el mayor uso de aleaciones basadas en Ti en aplicaciones estructurales de alta temperatura. La demanda de aumentar la temperatura de servicio de estas aleaciones más allá de 550 °C (el límite de temperatura típico) requiere un estudio cuidadoso para comprender el papel que tiene la composición en el comportamiento de oxidación de las aleaciones basadas en Ti [1-3]. El intento de superar esta limitación en las aleaciones basadas en Ti ha llevado a la producción de aleaciones con una resistencia a la oxidación sustancialmente mejorada, como el β-21S, y también al desarrollo de recubrimientos y técnicas de preoxidación [1,4–6]. Si bien es tentador extrapolar el comportamiento de oxidación (por ejemplo, ley de la velocidad de oxidación, profundidad del ingreso de oxígeno y espesor de las incrustaciones) observado para un número limitado de composiciones bajo una determinada condición de oxidación a un rango de composición más amplio, existen numerosos ejemplos en la literatura donde las desviaciones de las relaciones esperadas se observan [7,8].'
# Procesar el texto con Spacy

#texto="aleaciones con resistencia a la oxidación sustancialmente mejorada"
doc = nlp(texto)

# Extraer los noun chunks

lis= []


for chunk in doc.noun_chunks:
    #print(chunk.text)
    #print(chunk.end)
    #print(chunk.start)
    #print(chunk.text, chunk.root.text)
    if len(chunk.text) <2:
        continue
    print(chunk.text, chunk.start, chunk.end)
    #print("Inicio del chunk:", chunk.start_char)
    #print("Fin del chunk:", chunk.end_char)
    lis.append((chunk.start_char,chunk.end_char))


print('ZZZ>>>>>>>>>')

lis2=[]

def res(tokens,labels):


    candiate=''
    essential = {'NOUN', 'PROPN'}
    possible = {'NOUN', 'ADJ', 'ADV', 'PROPN'}
    internals = {'DET', 'ADJ', 'ADV','ADP', 'PROPN', 'NOUN'}
    essentialbool=False

    for tok,lab in zip(tokens,labels):
        if lab in essential:
            candiate=candiate+tok+' '
            essentialbool=True
            continue

        if lab in possible:
            candiate = candiate + tok + ' '
            continue
        if lab in internals and essentialbool:
            candiate=candiate+tok+' '
        if lab not in internals and essentialbool:
            break
    return candiate.strip()


def res2(tokens, labels):
    candidates = []
    candidate = ''
    essential = {'NOUN', 'PROPN'}
    possible = {'NOUN', 'ADJ', 'ADV', 'PROPN'}
    internals = {'DET', 'ADJ', 'ADV', 'PROPN', 'NOUN','ADP'}
    not_possible_end={'ADP','DET'}
    last_label=''
    essentialbool = False

    for tok, lab in zip(tokens, labels):

        if lab == 'ADP' and tok=='para':
            lab='ADP2'

        if lab == 'PROPN' and len(tok) ==1:
            continue

        #print('candidate',candidate)
        #print(candidates)
        if lab in essential:
            candidate = candidate + tok + ' '
            essentialbool = True
            last_label = lab
            continue

        if lab in possible:
            candidate = candidate + tok + ' '
            last_label = lab
            continue
        if lab in internals and essentialbool:
            candidate = candidate + tok + ' '
            last_label = lab
            continue


        if lab not in internals and essentialbool:
            if last_label in not_possible_end:
                candidate= ' '.join(candidate.split()[:-1]).strip()
            last_label=''
            if len(candidate)<3 :
                continue
            candidates.append(candidate.strip())
            candidate = ''
            essentialbool=False
            continue
        last_label = ''
        candidate = ''
        essentialbool = False
    if len(candidate)>0:
        candidates.append(candidate)

    return candidates
tokens=[]
labels=[]
printed=[]
for token in doc:
    printed.append(str(token.text)+'-'+token.pos_)
    tokens.append(token.text)
    labels.append(token.pos_)

print(tokens,labels)
print(printed)
resu= res2(tokens, labels)
print(resu)

'''
for a,b in lis:
    #print(a,b)
    tokens=[]
    labels=[]
    for token in doc:
        #print(token.idx)
        if a <= token.idx <= b:
            #print(token.text, "-", token.pos_,token.idx)
            tokens.append(token.text)
            labels.append(token.pos_)

    print('-----')
    print(tokens)
    print(labels)
    print(res(tokens,labels))
'''
'''
for token in doc:
    print(token.pos_)
    print(token)
    print(token.pos)
    print(token.idx)

'''





from nltk.tokenize import sent_tokenize


'''
texto = "Hola, esto es una prueba. ¿Funcionará el tokenizador de frases? Espero que sí."

frases = sent_tokenize(texto, language='spanish')

for frase in frases:
    print(frase)
'''
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = TFBertModel.from_pretrained("bert-base-uncased")




#tokenizer = RobertaTokenizer.from_pretrained(modelname)
#model = RobertaModel.from_pretrained(modelname, output_attentions=True)

'''
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModel.from_pretrained(modelname, output_attentions=True)
encoded_input = tokenizer(["ti-based alloys"], return_tensors='pt')

print(encoded_input)
output = model(**encoded_input)
print(output.attentions)
'''