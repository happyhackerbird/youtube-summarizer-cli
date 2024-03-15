from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os

load_dotenv()
api_key = os.environ["PPLX_API_KEY"]

url = "https://www.youtube.com/watch?v=TwDJhUJL-5o"
loader = YoutubeLoader.from_youtube_url(
    url, add_video_info=False
)
docs = loader.load()

# Define prompt
prompt_template = """Write a summary of the following video transcript. Highlight key points of the video and include the timestamp in (mm:ss) format.
"{text}"
VIDEO SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
# llm = ChatPerplexity(temperature=0, pplx_api_key="Bearer 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", model="pplx-70b-online")
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

docs = loader.load()
print(stuff_chain.invoke(docs))