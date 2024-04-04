
import spacy






"""## VERSION PABLO
    candidates=[]
    for chunk in doc.noun_chunks:
        #print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
        chunk_lower=remove_starting_articles(chunk.text.lower())
        candidates.append(chunk_lower)

    ## VERSION CEJAS
    '''
    tagged = []
    for token in doc:
        tagged.append((str(token), token.tag_))
    tagged = [tagged]
    print('tag')
    print(tagged)
    text_obj = InputTextObj(tagged, 'en')
    candidates = extract_candidates(text_obj)
    # print(candidates)
    '''"""