
import re
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy

class NounPhrasesIdentifier():

    def __init__(self,lang):
        self.lang=lang
        if lang == 'es':
            self.nlp = spacy.load("es_core_news_sm")
            print('Spanish Model')
        else:
            self.nlp = spacy.load("en_core_web_sm")

    def remove_starting_articles(self,text):
            # Lista de artículos a eliminar

            if self.lang == 'es':
                articles = ['se ', 'la ', 'el ', 'un ', 'una ', 'unos ', 'unas ', 'los ', 'las ', 'esta ', 'este ',
                            'estos ', 'estas ', 'cada ']
            else:
                articles = ['a ', 'the ', 'an ', 'this ', 'those ', 'that ', 'which ', 'every ']
            text_low = text

            # Iterar sobre cada artículo
            for article in articles:
                # Si el texto comienza con el artículo, quitarlo
                if text_low.lower().startswith(article):
                    text = text[len(article):]  # Quitar el artículo

            return text

    def generate_candidates(self, text, language):

        if self.lang=='es':
            candidates=[]
            doc = self.nlp(text)
            lis=[]
            for chunk in doc.noun_chunks:
                if len(chunk.text) < 2:
                    continue
                lis.append((chunk.start_char, chunk.end_char))
            for a, b in lis:
                # print(a,b)
                tokens = []
                labels = []
                for token in doc:
                    # print(token.idx)
                    if a <= token.idx <= b:
                        # print(token.text, "-", token.pos_,token.idx)
                        tokens.append(token.text)
                        labels.append(token.pos_)


                candidate= self.construct_candidates_es(tokens, labels)
                if len(candidate) > 1:
                    candidates.append(candidate)








        else:
            candidates = []
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                # chunk.root.dep_, chunk.root.head.text)
                chunk_processed = self.remove_starting_articles(chunk.text.lower())
                # chunk_processed = chunk_processed.lower()
                if len(chunk_processed) < 2:
                    continue
                candidates.append(chunk_processed)
        return candidates



    def construct_candidates_es(self, tokens, labels):

        candiate = ''
        essential = {'NOUN', 'PROPN'}
        possible = {'NOUN', 'ADJ', 'ADV', 'PROPN'}
        internals = {'DET', 'ADJ', 'ADV', 'ADP', 'PROPN', 'NOUN'}
        essentialbool = False

        for tok, lab in zip(tokens, labels):
            if lab in essential:
                candiate = candiate + tok + ' '
                essentialbool = True
                continue

            if lab in possible:
                candiate = candiate + tok + ' '
                continue
            if lab in internals and essentialbool:
                candiate = candiate + tok + ' '
            if lab not in internals and essentialbool:
                break
        return candiate.strip()





class ModelEmbedding():

    def __init__(self, model_name, type, tokenizer, model):
        self.extractor = pipeline(model=model_name, task="feature-extraction")
        self.tokenizer = tokenizer
        self.model= model
        self.type = type

    def __call__(self, data, oov_way='avg', filter_spec_tokens=True):
        #print(data)
        if self.type=='roberta':
            if len(data)>0:
                data[0]= ' '+data[0].lower()
        result = self.extractor(data, return_tensors=True,truncation=True)


        ids = self.tokenizer(data)
        lis = []
        for res, input in zip(result, ids['input_ids']):
            lis.append((input, res[0].cpu().detach().numpy()))

        if self.type == 'bert':
            return self.embedding_constructor_bert(lis)
        if self.type == 'roberta':
            return self.embedding_constructor_roberta(lis)

        return self.embedding_constructor_bert(lis)

    def embedding_constructor_roberta(self, batches, oov_way='avg', filter_spec_tokens=True):
        #print("ROBERTA")
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.
        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.
        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        # for token_ids, sequence_outputs in batches:
        for batch in batches:
            tokens = []
            tensors = []
            oov_len = 1

            for token_id, sequence_output in zip(batch[0], batch[1]):

                token = self.tokenizer.decode(token_id)
                #print(token)
                if token == '[PAD]':
                    # [PAD] token, sequence is finished.
                    break

                if token == '<pad>':
                    # [PAD] token, sequence is finished.
                    break
                if token == '[CLS]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '[SEP]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '</sep>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '<sep>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue

                if token == '<s>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '</s>' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token.startswith(' '):
                    token = token[1:]
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)

                else:  # iv, avg last oov
                    if len(tokens)==0:
                        tokens.append(token)
                        tensors.append(sequence_output)
                        continue

                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1

            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences

    def embedding_constructor_bert(self, batches, oov_way='avg', filter_spec_tokens=True):
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.
        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words
        filter_spec_tokens : bool
            filter [CLS], [SEP] tokens.
        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        # for token_ids, sequence_outputs in batches:
        for batch in batches:
            tokens = []
            tensors = []
            oov_len = 1

            for token_id, sequence_output in zip(batch[0], batch[1]):

                token = self.tokenizer.decode(token_id)

                if token == '[PAD]':
                    # [PAD] token, sequence is finished.
                    break
                if token == '[CLS]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token == '[SEP]' and filter_spec_tokens:
                    # [CLS], [SEP]
                    continue
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences


    def getAttentions(self,sentence):
        if self.type == 'roberta':
            sentence = separar_caracteres(sentence)
            encoded_input = self.tokenizer(' ' + sentence.lower(), return_tensors='pt',truncation=True)
            output = self.model(**encoded_input)
            attentions = output.attentions

        else:
            encoded_input = self.tokenizer(sentence, return_tensors='tf')
            output = self.model(encoded_input, output_attentions=True)
            attentions = output.attentions
        return attentions,encoded_input

    def get_tokens(self,line):
        if self.type == 'roberta':
            tokens = self.tokenizer.tokenize(' '+line)
        else:
            tokens = self.tokenizer.tokenize(line)
        return tokens


import re
def separar_caracteres(texto):
    # Utilizamos una expresión regular para identificar los paréntesis pegados al texto
    patron = r'(\()|(\))|(-)(\w)'
    texto_separado = re.sub(patron, r'\1 \2\3 \4', texto)
    return texto_separado


# Ejemplo de uso
texto_original = '(ejemplo-texto)'
texto_separado = separar_caracteres(texto_original)
print("Texto original:", texto_original)
print("Texto separado:", texto_separado)




