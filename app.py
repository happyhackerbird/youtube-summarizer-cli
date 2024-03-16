from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


def get_youtube_transcript(url):
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=False
    )
    docs = loader.load()
    return docs

def get_prompt():
    prompt_template = """Write a summary of the video transcript delimited in <>. Proceed section by section of the video, one paragraph for each section. State the key points of the section (include the timestamp in (mm:ss) format). 
    <{text}>
    VIDEO SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt


def get_summary_chain(model="mixtral-8x7b-instruct"):
    llm = ChatPerplexity(temperature=0, model=model)
    llm_chain = LLMChain(llm=llm, prompt=get_prompt())
    # Use StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain

if __name__ == '__main__':
    load_dotenv()
    url = "https://www.youtube.com/watch?v=TwDJhUJL-5o"
    docs = get_youtube_transcript(url)
    summary_chain = get_summary_chain("mixtral-8x7b-instruct")
    response = summary_chain.invoke(docs)
    print(response['output_text'])
