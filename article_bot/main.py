from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import re

keyword = "<input_your_own_keyword/keywords>"
os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"


def get_links(keyword):
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search.run(tool_input=keyword)

    links = []
    parsed_links = re.findall(r'link:\s*(https?://[^\],\s]+)', results)
    
    for link in parsed_links:
        links.append(link)
        
    return links