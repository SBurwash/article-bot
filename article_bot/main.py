from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import bs4
import re
from dotenv import load_dotenv
import os

load_dotenv()
keyword = "Trump, Tarrifs, Pandas"



def get_links(keyword):
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    results = search.run(tool_input=keyword)

    links = []
    parsed_links = re.findall(r'link:\s*(https?://[^\],\s]+)', results)
    
    for link in parsed_links:
        links.append(link)
        
    return links

def save_file(content, filename):
    directory = "blogs"
            
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)

    with open(filepath, 'w') as f:
        f.write(content)
        print(f" 🥳 File saved as {filepath}")

bs4_strainer = bs4.SoupStrainer(('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'))
document_loader = WebBaseLoader(web_path=(get_links(keyword)))
docs = document_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True,)
splits = splitter.split_documents(docs)
vector_store = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
retriever = vector_store.as_retriever(search_type="similarity", search_kwards={"k": 10})
template = """
Given the following information, generate a blog post
Write a full blog post that will rank for the following keywords: {keyword}
                
Instructions:
The blog should be properly and beautifully formatted using markdown.
The blog title should be SEO optimized.

The blog title, should be crafted with the keyword in mind and should be catchy and engaging. But not overly expressive.
Each sub-section should have at least 3 paragraphs.
Each section should have at least three subsections.
Sub-section headings should be clearly marked.
Clearly indicate the title, headings, and sub-headings using markdown.
Each section should cover the specific aspects as outlined.
For each section, generate detailed content that aligns with the provided subtopics. Ensure that the content is informative and covers the key points.
Ensure that the content flows logically from one section to another, maintaining coherence and readability.
Where applicable, include examples, case studies, or insights that can provide a deeper understanding of the topic.
Always include discussions on ethical considerations, especially in sections dealing with data privacy, bias, and responsible use. Only add this where it is applicable.
In the final section, provide a forward-looking perspective on the topic and a conclusion.
Please ensure proper and standard markdown formatting always.
Make the blog post sound as human and as engaging as possible, add real world examples and make it as informative as possible.
You are a professional blog post writer and SEO expert.

Context: {context}
Blog: 
"""
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
prompt = PromptTemplate.from_template(template=template)
# print("".join(doc.page_content for doc in docs))
# print({"context": "".join(doc.page_content for doc in docs)})

def format_docs(docs):
    return "".join([d.page_content for d in docs])

chain = (
    # {"context": retriever | "".join(doc.page_content for doc in docs), "keyword": RunnablePassthrough()}
    {"context": retriever | format_docs, "keyword": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke(input=keyword)

print(response)

save_file(content=response, filename=keyword+".md")