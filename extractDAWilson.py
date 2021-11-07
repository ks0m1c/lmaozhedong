# Adapted from https://github.com/epikulski/digitalarchive 
# install digitalarchive : pip install digitalarchive
# match docs https://digitalarchive.readthedocs.io/en/latest/api.html#digitalarchive.models.Document.match

import digitalarchive
from digitalarchive.models import Language
import pdfkit
import os

TRANSLATION_DIR = "translations"
TRANSLATION_HTML_DIR = "html"
TRANSLATION_PDF_DIR = "pdf"

def downloadMediaFile(doc, name):
    media_files = doc.media_files
    for i, media_file in enumerate(media_files):
        media_file.hydrate()
        print(media_file.content_type)

        with open("media_files/{}_{}.pdf".format(name, i), "wb") as f:
            f.write(media_file.pdf)

def getTranslations(doc, name):
    translations = doc.translations
    if len(translations)>0:
        for i, translation in enumerate(translations):
            translation.hydrate()
            if translation.pdf is not None:
                print("translation not none pdf")
                with open("translations/{}_{}.pdf".format(name, i), "wb") as f:
                    f.write(translation.pdf)
            print(translation.html)
            if translation.raw is not None:
                with open("translations/html/raw_{}_{}.html".format(name, i), "wb") as f:
                    f.write(translation.raw)

def getTranscripts(doc, name):
    transcripts = doc.transcripts
    if len(transcripts)>0:
        print("has transcripts")
        for i, transcript in enumerate(transcripts):
            transcript.hydrate()
            if transcript.pdf is not None:
                print("transcripts not none pdf")
                with open("transcripts/{}_{}.pdf".format(name, i), "wb") as f:
                    f.write(transcript.pdf)


def getTranslationHtmlToPdf():
    html_files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(TRANSLATION_DIR, TRANSLATION_HTML_DIR)):
        html_files = filenames
        break
    for html_file in html_files:
        path = os.path.join("./",TRANSLATION_DIR, TRANSLATION_HTML_DIR, html_file)
        file_name = html_file.split(".")[0]
        pdfkit.from_file(path, os.path.join("./",TRANSLATION_DIR, TRANSLATION_PDF_DIR, '{}.pdf'.format(file_name)))


def main(docs_limit = None):
    korean_war_docs = digitalarchive.Document.match(description ="korean war telegram").all()
    print(len(korean_war_docs))
    if docs_limit is not None:
        korean_war_docs = korean_war_docs[:docs_limit]
    print("-----------------------")
    for i, doc in enumerate(korean_war_docs):
        doc.hydrate()
        print(doc)
        title = doc.title.replace(" ", "_")
        title = doc.title.replace(".", "_")
        downloadMediaFile(doc, title)
        getTranslations(doc,title)
        getTranscripts(doc,title)
    getTranslationHtmlToPdf()

main()
