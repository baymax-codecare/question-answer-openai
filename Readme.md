This app talks with multiple documents like pdf, docx, txt files.
This app takes two parts:
- chat_ingest.py

This app ingests document files into chroma db.
For this, tesseract engine and poppler are used for converting pdf to text.
You can see global parameters in config.py.
You have to put all documents in DOC_FOLDER_PATH.

- chat.py
Question-Answering System, which would talk with ingested documents.
It runs on Gradle.
If you write a question, then relevant answers will be shown.