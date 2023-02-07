import streamlit as st
import os
from haystack.utils import fetch_archive_from_http, clean_wiki_text, convert_files_to_docs
from haystack.schema import Answer
from haystack.utils import print_documents
from haystack.document_stores import InMemoryDocumentStore,FAISSDocumentStore
from haystack.pipelines import ExtractiveQAPipeline,DocumentSearchPipeline
from haystack.nodes import FARMReader, TfidfRetriever,EmbeddingRetriever
import logging
import pandas as pd
from markdown import markdown
from annotated_text import annotation
from PIL import Image

os.environ['TOKENIZERS_PARALLELISM'] ="false"
def get_results(query, retriever, n_docs = 15):
  return [(item.content, item.to_dict()['meta']) for item in retriever.retrieve(query, top_k = n_docs)]

def load_dataset():
    df = pd.read_csv('Scrapped_data.csv')
    df = df.drop("Unnamed: 0", axis=1)
    x = df[['url', 'article']].rename(columns={ 'url' : 'author','article':'content'}).to_dict(orient='records')
    return x



#Haystack Components
@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},allow_output_mutation=True)
def start_haystack():

    document_store = FAISSDocumentStore(sql_url = "sqlite:///faiss_document_store_5.db")
    df_dict = load_dataset()
    load_and_write_data(df_dict,document_store)
    #
    #  multi-qa-mpnet-base-dot-v1
    retriever = EmbeddingRetriever(embedding_model='sentence-transformers/all-mpnet-base-v2',document_store = document_store,model_format='sentence_transformers')
    document_store.update_embeddings(retriever=retriever)

    #reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2-distilled", use_gpu=True)
    #pipeline = ExtractiveQAPipeline(retriever)
    #print(retriever)
    #pipeline = DocumentSearchPipeline(retriever)

    #query = "Tell me something about that time when they play chess."
    #result = pipeline.run(query, params={"Retriever": {"top_k": 2}})
    return retriever

def load_and_write_data(df_dict,document_store):
    #doc_dir = './article_txt_got'
    #docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    document_store.write_documents(df_dict)

retriever = start_haystack()

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

set_state_if_absent("question", "Canto CEO")
set_state_if_absent("results", None)


def reset_results(*args):
    st.session_state.results = None

#Streamlit App

image = Image.open('got-haystack.png')
st.image(image)

st.markdown( """
 # This is a testing env for the BlackRay document retrieval system. Feel free to test out various queries regarding content in the fillowing companies - 
## Canto, Bynder, Gumlet, Imagekit, Aha, Hygraph, Yext, Lucidworks
""", unsafe_allow_html=True)

question = st.text_input("", value=st.session_state.question, max_chars=100, on_change=reset_results)

def ask_question(question):
    #prediction = pipeline.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
    #res = get_results(question, retriever)
    #res = get_results(q, retriever_faiss)

    prediction = get_results(question, retriever)
    #prediction = pipeline.run(query=question, params={"Retriever": {"top_k": 10}})
    #print(prediction)
    #print_documents(prediction, max_text_len=100, print_name=True, print_meta=True)
    results = []
    for result in prediction:
        dct = {
            "Text": result[0],
            "Name" : result[1]
        }
        results.append(dct)
    #print(results)
    return results


if question:
    with st.spinner("👑 &nbsp;&nbsp; Performing semantic search on royal scripts..."):
        try:
            msg = 'Asked ' + question
            logging.info(msg)
            st.session_state.results = ask_question(question)    
        except Exception as e:
            logging.exception(e)
    


if st.session_state.results:
    st.write('## Top Results')
    for count, result in enumerate(st.session_state.results):
        if result["Text"]:
            answer, context = result["Text"], result["Name"]['author']
            st.write(context)
            st.write(answer)
            #start_idx = context.find(answer)
            #end_idx = start_idx + len(answer)
            #st.write(
            #    markdown(context[:start_idx] + str(annotation(body=answer, label="ANSWER", background="#964448", color='#ffffff')) + context[end_idx:]),
            #    unsafe_allow_html=True,
            #)
            #st.markdown(f"**Relevance:** {result['relevance']}")
        else:
            st.info(
                "🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!"
            )

