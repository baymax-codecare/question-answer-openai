from config import *
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import gradio as gr
from langchain.prompts import PromptTemplate

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"],
    template=template
)

def get_answer(question):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # index = pinecone.Index(PINECONE_INDEX_NAME)
    # vectorStore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings, "text", PINECONE_NAMESPACE)
    # docs = vectorStore.similarity_search(question)

    vectordb = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)
    docs = vectordb.similarity_search(question)
    #print(docs)

    qa = load_qa_chain(OpenAI(temperature=OPENAI_TEMPERATURE, openai_api_key=OPENAI_API_KEY), chain_type="stuff", memory=memory, prompt=prompt)
    result = qa({"input_documents": docs, "human_input": question}, return_only_outputs=True)
    #print(result['output_text'])

    return result['output_text']

# get_answer('How to simplify the usage of the API?')
# get_answer('What does Climate imply?')
# # get_answer('I want to know what the maxillary sinus is.')
# get_answer('What was it I wanted to know about again?')
# get_answer('What did I ask just?')
# get_answer('What does Poppler have?')
#
# print(memory)

demo = gr.Interface(fn=get_answer, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(share=True)