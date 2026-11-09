def get_embedding_model(config):
    embedding_type = config['embedding_model']
    if embedding_type == 'OpenAI':
        from .openai import OpenAIEmbedding
        return OpenAIEmbedding(config)
    elif embedding_type == 'Cohere':
        from .cohere import CohereEmbedding
        return CohereEmbedding(config)
    elif embedding_type == 'Google Gemini':
        from .google_gemini import GoogleGeminiEmbedding
        return GoogleGeminiEmbedding(config)
    elif embedding_type == 'Azure OpenAI':
        from .azure_openai import AzureOpenAIEmbedding
        return AzureOpenAIEmbedding(config)
    elif embedding_type == 'Hugging Face':
        from .huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(config)
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_type}")
