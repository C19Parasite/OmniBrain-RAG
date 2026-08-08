DOCUMENTS = []

def add_document(filename: str, text: str):
    DOCUMENTS.append({
        "filename": filename,
        "text": text
    })

def get_documents():
    return DOCUMENTS