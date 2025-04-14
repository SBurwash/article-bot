from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
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