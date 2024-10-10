import os, json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain_community.llms import huggingface_endpoint
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from termcolor import cprint
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from pymongo.mongo_client import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import certifi
from langchain.globals import set_debug
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_headers=['*'],
        allow_methods=["*"],
    )
]

app = FastAPI(middleware=middleware)

# set_debug(True)
try:
    load_dotenv('.env')

    # region mongo stuff

    connection_str = os.getenv('MONGO_URL')
  
    mongo_uri = connection_str
    COLLECTION_NAME = 'Perf'

    ca = certifi.where()

    db_client = MongoClient(mongo_uri, tlsCAFile=ca) #'mongodb://localhost:27017'
except Exception as ex:
    print(ex)

try:
    db_client.admin.command('ping')
    print('Mongodb connection successful!')
except Exception as ex:
    print(ex)

db = db_client.perf
formula_collection = db.formulas

#endregion

llm: huggingface_endpoint = None
conversation: ConversationalRetrievalChain = None
chat_history:list = []
vector_store: faiss.FAISS = None
smell_parser: StructuredOutputParser = None
material_parser: StructuredOutputParser = None
formula_chain:RetrievalQA = None

# region response parsers

class SmellMaterials(BaseModel):
    materials:list = Field(description='answer as a python string list of materials')

class MaterialIdentifier(BaseModel):
    question:str = Field(description='user question')
    materials:list = Field(description='answer as a python string list of materials')

smell_schema = ResponseSchema(
    name='smells',
    description='a list of materials which make up the smell in question.'
)

smell_parser = StructuredOutputParser.from_response_schemas([smell_schema])

material_schema = ResponseSchema(
    name='materials',
    description='the materials passed to the query as a python list. DO NOT ANSWER THE QUESTION.'
)

material_parser = StructuredOutputParser.from_response_schemas([material_schema])

#endregion

def init():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    store_path = os.path.dirname(__file__)
    store_path = os.path.join(store_path, 'vector_store')

    global vector_store
    vector_store = faiss.FAISS.load_local(
        folder_path=store_path, 
        allow_dangerous_deserialization=True, 
        embeddings=embeddings, 
        index_name='perf-docs-1k')

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    HUGGINGFACEHUB_API_TOKEN='tkn'

    global llm
    llm = huggingface_endpoint.HuggingFaceEndpoint(repo_id=repo_id, temperature=0.03, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

    custom_template = """[INST] You are master perfumer. Your primary task is to ONLY answer questions about perfumery. 
        If the question is not about perfumery or smell, do not answer it.
        If the context is not relevant, please answer the question by using your own knowledge about the topic.

        Rules: 
        1.If the user greets you, greet them back.
        2.DO Not cite the sources.
        3.DO NOT add extra information aside from the answer.
        4.If there is no chat history, DO NOT make up a chat history.
        5.DO NOT speculate. 
        6.If you don't know the answer, don't make up answers or speculate. 
        7.Keep the answer short.
        9.Use a combination of the context and your own knowledge.

        Chat History: {chat_history}

        Context: {context}

        Human: Don't justify your answers. Don't give information not related to perfumery or chat history.
        
        AI: Sure! I will stick to all the information about perfumery. 

        Human: Hello.

        AI: Hi! Do you have any questions about perfumery which I can answer?
        
        Human: {question}
        
        AI: [/INST]
        """

    PROMPT = PromptTemplate(
        input_variables=["question", "chat_history", "context", "answer"], 
        template=custom_template
    )

    memory_buffer = ConversationBufferMemory(memory_key='chat_history', input_key='question', output_key='answer', return_messages=True)

    global conversation
    conversation = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                        retriever=vector_store.as_retriever(), 
                                                        chain_type='stuff', 
                                                        combine_docs_chain_kwargs={"prompt":PROMPT})
    conversation.memory = memory_buffer

def smell_search(smell:str, parser):
    # identify materials by smell ##########################
    qa_prompt = PromptTemplate(
        template="""Provide a list of the 10 most common aroma chemicals which contribute to the smell being asked about. 
        
        Query: {question}
        
        {format_instructions}
        """,
        input_variables=['question'],
        partial_variables={'format_instructions': parser.get_format_instructions()}
    )

    qa_chain = qa_prompt | llm | smell_parser

    try:
        materials = qa_chain.invoke({'question':f'what chemicals make up the smell of {smell}?'})
    except Exception as ex:
        return ex
    
    return materials

def material_search(materials:list, parser):
    formulas = []
    return_val = None

    for m in materials:
        # found = formula_collection.find({"$text": {"$search": m}})
        found = formula_collection.aggregate([
            {
                '$search': {
                    'index': 'default',
                    'text': {
                        'query': m,
                        'path': {
                            'wildcard': '*'
                        }
                    }
                }
            },
            {
                '$project': {
                    '_id': 1,
                    'formula_name': 2,
                    'formula_items': 3,
                    'score': {'$meta': 'searchScore'}
                }
            }
        ])
    
        for f in found:
            formulas.append(f)

    returns = []
    if len(formulas) > 0:
        for f in formulas:
            returns.append(json.dumps({'id': str(f['_id']), 'name': f['formula_name'], 'formulaitems': f['formula_items'], 'score': f['score']}))
    else:
        return_val = 'No matching formulas found.'

    return_val = returns
    return return_val

def answer_query(query:str):
    global chat_history

    answer = ''

    if query.find('smell:') > -1:
        # qa_prompt = PromptTemplate(
        #     template="""List only the smells in the query. 
        #     DO NOT ANSWER THE QUERY. 
        #     DO NOT ADD ANY EXTRA INFORMATION.
            
        #     query: {question}
            
        #     {format_instructions}
        #     """,
        #     input_variables=['question'],
        #     partial_variables={'format_instructions': smell_parser.get_format_instructions()}
        # )

        # qa_chain = qa_prompt | llm | smell_parser

        # smells = qa_chain.invoke({'question': query})

        smell = query.split(':')[1]

        answer = smell_search(smell, smell_parser)
    elif query.find('material:') > -1:
        # qa_prompt = PromptTemplate(
        #     template="""This is in the material category. Provide only the exact materials being asked about
        #     as a python list.
        #     Take note of spaces in the names.
        #     DO NOT ANSWER THE QUERY.
        #     DO NOT ADD ANY EXTRA INFORMATION.

        #     query: {question}
            
        #     {format_instructions}
        #     """,
        #     input_variables=['question'],
        #     partial_variables={'format_instructions': material_parser.get_format_instructions()}
        # )

        # qa_chain = qa_prompt | llm | material_parser

        # materials = qa_chain.invoke({'question': query})['materials']
        materials = query.split(':')[1]
        materials = materials.split(',')
        answer = material_search(materials, material_parser)
    else:
        # determine relevance
        # is_relevant = get_relevance(query)

        # if is_relevant == 'False':
        #     answer = "I'm sorry, but I can only answer questions about perfumery."
        #     return answer

        docs = vector_store.similarity_search(query, k=4)
      
        answer =  conversation.invoke(input={'question': query, 'chat_history': chat_history, 'context': docs})

        chat_history = answer.get('chat_history')
        answer = answer['answer']

    return answer

def get_relevance(text):
    # qa_prompt = PromptTemplate(
    #     template="""Your task is to determine the relevance of the question to perfumery.
    #     The only relevant questions are regarding:
    #     -aroma chemicals 
    #     -perfumery formulas 
    #     -perfumery creation
    #     -questions about perfumery smells
    #     -greetings
    #     -conversation history

    #     DO NOT answer the question.
    #     Format the answer only as 'True' or 'False'.

    #     Input: {input}
    #     Answer:
    #     """,
    #     input_variables=['input']
    # )

    qa_prompt = PromptTemplate(
        template="""Is the input question related to perfumery or smells?
        DO NOT answer the question. Return only boolean 'True' or 'False'.

        Input: {input}
        Answer:
        """,
        input_variables=['input']
    )

    qa_chain = qa_prompt | llm 

    is_relevant = qa_chain.invoke({'input': text}).strip()

    return is_relevant
    
def get_formula_by_id(id:str):
    formula = formula_collection.find_one({'_id': ObjectId(id)})
    return formula

def run_loop():
    while True:
        print(end='\n')
        cprint('question:  ', 'yellow', end=' ')
        prompt = input()
        answer = answer_query(prompt)
        cprint(f'answer:  {answer}', 'light_cyan', end=' ')

#region endpoints

@app.get("/answer_query/")   
async def Question(query):
    return answer_query(query)


#endregion

init()
# run_loop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
