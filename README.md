# AttentionRankLib
Repository to develop AttentionRank algorithm as library

Based on the work: https://github.com/hd10-iupui/AttentionRank

  


## Install
Using Python 3.9


```
pip install -r requirements.txt
```


```
pip install -e .
```

```
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

```


## Execution

```
python main.py --dataset_name example --model_name_or_path PlanTL-GOB-ES/roberta-base-bne --model_type roberta --lang es --type_execution exec --k_value 15
```
Important: The dataset must be in this folder with any name and all documents must be inside another folder named docsutf8
Results will be provided inside the folder of the dataset in a folder named res+k_value

## Docker run 
For a fast run use the dockerfile and this two commands. 

```
docker build -t attentionranklib .

``` 

```
docker run --rm -v ./example:/app/example attentionranklib --dataset_name example --model_name_or_path PlanTL-GOB-ES/roberta-base-bne --model_type roberta --lang es --type_execution exec --k_value 15
```



## Acknowledgments 
Para su desarrollo este código ha recibido financiación del proyecto INESData (Infraestructura para la INvestigación de ESpacios de DAtos distribuidos en UPM), un proyecto financiado en el contexto de la convocatoria UNICO I+D CLOUD del Ministerio para la Transformación Digital y de la Función Pública en el marco del PRTR financiado por Unión Europea (NextGenerationEU).

Este código se ha mejorado y adaptado en el marco del proyecto TeresIA, proyecto de investigación financiado con fondos de la Unión Europea Next GenerationEU / PRTR a través del Ministerio de Asuntos Económicos y Transformación Digital (hoy Ministerio para la Transformación Digital y de la Función Pública). 

## Paper Citation

```bibtext
@inproceedings{Calleja2024,
  author    = {Pablo Calleja and Patricia Martín-Chozas and Elena Montiel-Ponsoda},
  title     = {Benchmark for Automatic Keyword Extraction in Spanish: Datasets and Methods},
  booktitle = {Poster Proceedings of the 40th Annual Conference of the Spanish Association for Natural Language Processing 2024 (SEPLN-P 2024)},
  series    = {CEUR Workshop Proceedings},
  volume    = {3846},
  pages     = {132--141},
  year      = {2024},
  publisher = {CEUR-WS.org},
  address   = {Valladolid, Spain},
  month     = {September 24-27},
  urn       = {urn:nbn:de:0074-3846-7},
  url       = {https://ceur-ws.org/Vol-3846/}
}
```