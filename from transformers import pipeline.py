from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarize a lengthy article using a pre-trained model.
    
    Args:
        text (str): The full article text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
    
    Returns:
        str: A summarized version of the input text.
    """
    if len(text.split()) < 50:
        return "Text is too short to summarize meaningfully."

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example input: You can replace this with any lengthy article
article_text = """
The development of natural language processing (NLP) has revolutionized how we interact with machines.
By teaching computers to understand and generate human language, NLP powers technologies like chatbots, voice assistants,
language translation tools, and more. Recent advances, particularly in transformer-based architectures like BERT and GPT,
have drastically improved machine understanding of context and nuance in text.
These tools are now widely used in both research and commercial applications to automate text analysis, enhance user experience,
and create more natural interfaces between humans and machines.
"""

# Run the summarizer
summary = summarize_text(article_text)
print("📝 Original Text:\n", article_text)
print("\n🔍 Summary:\n", summary)
