def upload_file(file):
    """
    Mock upload API.
    This will be replaced with the real backend API later.
    """

    return {
        "success": True,
        "message": f"{file.name} uploaded successfully"
    }


def ask_question(question):
    """
    Mock chat API.
    This will be replaced with the real backend API later.
    """

    return {
        "success": True,
        "answer": f"Mock response for: {question}"
    }