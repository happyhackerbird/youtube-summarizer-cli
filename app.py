from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

import os
import typer
from typing import Optional


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


def get_summary_chain(model, key):
    if model=="gpt-4-turbo-preview":
        llm = ChatOpenAI(temperature=0, model=model, openai_api_key=key)
    else:
        llm = ChatPerplexity(temperature=0, model=model, pplx_api_key=key)
    llm_chain = LLMChain(llm=llm, prompt=get_prompt())
    # Use StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain

app = typer.Typer()

@app.command()
def main(model: str = typer.Option("mixtral-8x7b-instruct", help="The LLM model to use"),
          key: Optional[str] = typer.Option(None, help="The API key for the LLM model"),
          url: str = typer.Argument(help="The URL of the video to summarize")):
    """
    YouTube Summarizer using Perplexity AI & ChatGPT-4.
    """
    if model=="gpt-4":
        model="gpt-4-turbo-preview"
        key = key or os.getenv("OPENAI_API_KEY")
    else:
        key = key or os.getenv("PPLX_API_KEY")
    if key is None:
        typer.echo("Error: An API key is required.", err=True)
        raise typer.Abort()
    
    docs = get_youtube_transcript(url)
    summary_chain = get_summary_chain(model, key)
    response = summary_chain.invoke(docs)
    print(response['output_text'])


if __name__ == "__main__":
    app()