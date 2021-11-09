# Adapted from https://github.com/epikulski/digitalarchive 
# install digitalarchive : pip install digitalarchive
# match docs https://digitalarchive.readthedocs.io/en/latest/api.html#digitalarchive.models.Document.match

import digitalarchive
from digitalarchive.models import Language
import pdfkit
import os
from pathlib import Path

EXTRACTED_DATA_DIR = "extracted_data"
TRANSLATION_DIR = "translations"
HTML_DIR = "html"
PDF_DIR = "pdf"
MEDIA_FILES_DIR = "media_files"
TRANSCRIPTS_DIR = "transcripts"
TRANSLATION_DIR_PDF = "direct_pdf"
UNKNOWN_CREATOR = "unknown_creator"
UNKNOWN_DATE = "unknown_date"

ERRORS = 0

def downloadMediaFile(doc, creator, date, name):
    media_dir =  os.path.join(EXTRACTED_DATA_DIR, MEDIA_FILES_DIR, creator, date)

    media_files = doc.media_files
    for i, media_file in enumerate(media_files):
        Path(media_dir).mkdir(parents=True, exist_ok=True)
        media_file.hydrate()
        print(media_file.content_type)

        with open("{}/{}_{}.pdf".format(media_dir, name, i), "wb") as f:
            f.write(media_file.pdf)

def getTranslations(doc, creator, date, name):
    translation_dir =  os.path.join(EXTRACTED_DATA_DIR, TRANSLATION_DIR, creator, date)
    html_dir = os.path.join(translation_dir, HTML_DIR) 
    direct_pdf_dir = os.path.join(translation_dir, TRANSLATION_DIR_PDF) 

    translations = doc.translations
    if len(translations)>0:
        for i, translation in enumerate(translations):
            translation.hydrate()
            if translation.pdf is not None:
                Path(direct_pdf_dir).mkdir(parents=True, exist_ok=True)
                with open("{}/{}_{}.pdf".format(direct_pdf_dir, name, i), "wb") as f:
                    f.write(translation.pdf)
            if translation.raw is not None:
                Path(html_dir).mkdir(parents=True, exist_ok=True) 
                doc_name = "{}_{}".format(name, i)
                html_file = os.path.join(html_dir, "{}.html".format(doc_name))
                with open(html_file, "wb") as f:
                    f.write(translation.raw)
                print(html_file)
                pdf_dir = os.path.join(translation_dir, PDF_DIR)
                getHtmlToPdf(html_file, pdf_dir, doc_name)


def getTranscripts(doc, creator, date, name):
    transcripts_dir =  os.path.join(EXTRACTED_DATA_DIR, TRANSCRIPTS_DIR, creator, date)
    html_dir = os.path.join(transcripts_dir, HTML_DIR) 

    transcripts = doc.transcripts
    if len(transcripts)>0:
        print("has transcripts")
        for i, transcript in enumerate(transcripts):
            transcript.hydrate() 
            if transcript.pdf is not None:
                Path(transcripts_dir).mkdir(parents=True, exist_ok=True)
                print("transcripts not none pdf")
                with open("{}/{}_{}.pdf".format(transcripts_dir, name, i), "wb") as f:
                    f.write(transcript.pdf)

            if transcript.raw is not None:
                Path(html_dir).mkdir(parents=True, exist_ok=True) 
                doc_name = "{}_{}".format(name, i)
                html_file = os.path.join(html_dir, "{}.html".format(doc_name))
                with open(html_file, "wb") as f:
                    f.write(transcript.raw)
                print(html_file)
                pdf_dir = os.path.join(transcripts_dir, PDF_DIR)
                getHtmlToPdf(html_file, pdf_dir, doc_name)

def getHtmlToPdf(html_file, pdf_dir, doc_name):
    try:
        options = {"load-error-handling": "ignore", 'enable-local-file-access': None} 
        Path(pdf_dir).mkdir(parents=True, exist_ok=True)
        pdfkit.from_file(html_file, os.path.join(pdf_dir, '{}.pdf'.format(doc_name)), options=options)
    except: 
        global ERRORS
        ERRORS+=1
        print("could not get pdf for", doc_name)
    

def cleanFileName(file_name):
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(",", "_")
    file_name = file_name.replace(".", "_")
    file_name = file_name.replace("?", "_")
    file_name = file_name.replace("/", "_")
    file_name = file_name.replace("\r", "_")
    file_name = file_name.replace("\n", "_")
    file_name = file_name.replace(":", "_")
    file_name = file_name.replace("<", "_")
    file_name = file_name.replace(">", "_")
    file_name = file_name.replace("|", "_")
    file_name = file_name.replace("\\", "_")
    file_name = file_name.replace("'", "-")
    file_name = file_name.replace('"', "-")
    return file_name

def getDate(doc):
    try:
        date = doc.frontend_doc_date
        if date is None or not date:
            date = doc.date_range_start
        if date is None or not date:
            date = UNKNOWN_DATE
        
        return cleanFileName(str(date))
    except:
        return UNKNOWN_DATE

def removeEmptyDir(root_dir):
    num_removed = 0
    walk = list(os.walk(root_dir))
    for path, _, _ in walk:
        try:
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
                print("removed:", path)
                num_removed +=1
        except:
            print("could not remove", path)
    return num_removed


def main(docs_limit = None):
    korean_war_docs = digitalarchive.Document.match(description ="korean war telegram").all()
    print("total num documents", len(korean_war_docs))

    if docs_limit is not None:
        korean_war_docs = korean_war_docs[:docs_limit]
    
    print("-----------------------")
    hydration_errors = 0

    for i, doc in enumerate(korean_war_docs):
        print("-----DOCUMENT",i,"-------")
        try:
            doc.hydrate()
        except:
            print("could not hydrate", i)
            hydration_errors+=1
            continue
        
        # Using the creator's name as the directory to store their works
        try:
            creator_name = creator_name = doc.creators[0].name
            creator_name = cleanFileName(creator_name)
            
        except:
            creator_name = UNKNOWN_CREATOR
        
        date = getDate(doc)

        title = cleanFileName(doc.title)
        downloadMediaFile(doc, creator_name, date, title)
        getTranslations(doc,creator_name, date, title)
        getTranscripts(doc,creator_name, date, title)
    
    print("---------FINISHED EXTRACTING-----------")
    print("total num pdf errors", ERRORS)
    print("total num hydration errors", hydration_errors)
    print("total num documents", len(korean_war_docs))

    print("---------REMOVING EMPTY DIR-----------")
    num_removed = removeEmptyDir(EXTRACTED_DATA_DIR)
    print("removed", num_removed, "directories.")

main()