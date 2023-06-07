from config import *
from langchain.document_loaders import DirectoryLoader
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import glob, os
from os import listdir
from os.path import isfile, join
from pdf2image import convert_from_path
import pytesseract
import pypandoc
from langchain.document_loaders import PyPDFLoader

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def get_scanned_pdfs():
    print('Getting scanned pdf files...')
    pdfs = [f for f in listdir(DOC_FOLDER_PATH) if isfile(join(DOC_FOLDER_PATH, f)) and f.endswith(".pdf")]

    ret0 = []
    ret1 = []
    for pdf in pdfs:
        pdf_reader = PyPDF2.PdfReader(DOC_FOLDER_PATH + "/" + pdf)
        num_pages = len(pdf_reader.pages)
        total_chars = 0
        total_text_chars = 0
        for i in range(num_pages):
            page = pdf_reader.pages[i]
            text = page.extract_text()
            total_chars += len(text)
            total_text_chars += len(''.join([c for c in text if c.isalnum()]))

        if total_chars == 0:
            ret0.append(pdf)
        else:
            searchable_text_percentage = total_text_chars / total_chars * 100
            if searchable_text_percentage < 2:
                ret0.append(pdf)
            else:
                ret1.append(pdf)
    return ret0, ret1


def convert_scanned_pdfs2txt(files):
    print('Converting scanned pdf files to text...')
    for file in files:
        #print(file)
        pages = convert_from_path(pdf_path=DOC_FOLDER_PATH + "/" + file, dpi=350, poppler_path=POPPLER_PATH)
        i = 1
        images = []
        for page in pages:
            image_name = IMG_FOLDER_PATH + "/" + file + "_Page_" + str(i) + ".jpg"
            page.save(image_name, "JPEG")
            images.append(image_name)
            i = i + 1

        text = ''
        for image in images:
            text += pytesseract.image_to_string(image)
            text += "\n"

        with open(DOC_FOLDER_PATH + '/' + file + '_utf8.txt', 'w', encoding="utf-8") as f:
            f.write(text)

        os.rename(DOC_FOLDER_PATH + "/" + file, TMP_PDF_FOLDER_PATH + "/" + file)

def convert_textfile_to_utf8():
    print('Converting text files to utf8...')
    texts = [f for f in listdir(DOC_FOLDER_PATH) if isfile(join(DOC_FOLDER_PATH, f)) and f.endswith(".txt")]
    for textfile in texts:
        try:
            with open(DOC_FOLDER_PATH + '/' + textfile) as f:
                data = f.read()
            with open(DOC_FOLDER_PATH + '/' + textfile + '_utf8.txt', 'w', encoding='utf8') as f:
                f.write(data)
            os.rename(DOC_FOLDER_PATH + "/" + textfile, TMP_FOLDER_PATH + "/" + textfile)
        except:
            print(textfile)

def convert_doc2txt():
    print('Convert DOC files to Txt...')
    docs = [f for f in listdir(DOC_FOLDER_PATH) if isfile(join(DOC_FOLDER_PATH, f)) and f.endswith(".docx")]
    for doc in docs:
        pypandoc.convert_file(DOC_FOLDER_PATH + '/' + doc, 'plain', outputfile=DOC_FOLDER_PATH + '/' + doc + '.txt')
        os.rename(DOC_FOLDER_PATH + "/" + doc, TMP_DOC_FOLDER_PATH + "/" + doc)

def convert_pdfs2txt(files):
    print('Convert PDF files to Txt...')
    for file in files:
        loader = PyPDFLoader(DOC_FOLDER_PATH + "/" + file)
        pages = loader.load_and_split()
        content = ''
        for page in pages:
            content += page.page_content
        with open(DOC_FOLDER_PATH + '/' + file + '_utf8.txt', 'w', encoding='utf8') as f:
            f.write(content)
        os.rename(DOC_FOLDER_PATH + "/" + file, TMP_PDF_FOLDER_PATH + "/" + file)

def ingest():
    print('Ingest started...')
    loader = DirectoryLoader(DOC_FOLDER_PATH)
    rawDocs = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = textSplitter.split_documents(rawDocs)
    # len(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Pinecone-based Ingestion
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # index = pinecone.Index(PINECONE_INDEX_NAME)
    # Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME, namespace=PINECONE_NAMESPACE)

    # Chroma-based Ingestion
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIRECTORY)
    vectordb.persist()
    vectordb = None

    print('Ingest Done')

# convert_doc2txt()
# convert_textfile_to_utf8()
# pdfs_to_ocr, pdfs_to_txt = get_scanned_pdfs()
# convert_scanned_pdfs2txt(pdfs_to_ocr)
# convert_pdfs2txt(pdfs_to_txt)
ingest()