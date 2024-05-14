from langchain_community.embeddings import OllamaEmbeddings

embeddings=OllamaEmbeddings()

text="This is a sample documents."

query_result=embeddings.embed_query(text)
print("query-result",query_result[:5])



## using model = "llama2:7b"

embeddings=OllamaEmbeddings(model="llama2:7b")
query_result=embeddings.embed_query(text)
print("query-result using model - llama2:7b", query_result[:5])

