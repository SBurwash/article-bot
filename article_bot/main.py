from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import bs4
import re
from dotenv import load_dotenv
import os

load_dotenv()
keyword = "<input_your_own_keyword/keywords>"
print(os.environ['OPENAI_API_KEY'])



def get_links(keyword):
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search.run(tool_input=keyword)

    links = []
    parsed_links = re.findall(r'link:\s*(https?://[^\],\s]+)', results)
    
    for link in parsed_links:
        links.append(link)
        
    return links

bs4_strainer = bs4.SoupStrainer(('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'))
document_loader = WebBaseLoader(web_path=(get_links(keyword)))
docs = document_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True,)
splits = splitter.split_documents(docs)
vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
