import openai
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
import typer
from typing import Optional
from prompt_toolkit import prompt


def get_youtube_transcript(url):
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=False
    )
    docs = loader.load()
    return docs

def get_summary_prompt():
    prompt_template = """Write a summary of the video transcript delimited in <>. Proceed section by section of the video, one paragraph for each section. State the key points of the section (include the timestamp in (mm:ss) format). 
    <{text}>
    VIDEO SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt

def get_qa_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions about the video. Be sure to include timestamps in (mm:ss) with your answer.\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


def get_llm(model, key):
    if model=="gpt-4-turbo-preview":
        llm = ChatOpenAI(temperature=0, model=model, openai_api_key=key)
    else:
        llm = ChatPerplexity(temperature=0, model=model, pplx_api_key=key)
    return llm

def get_summary_chain(llm):
    llm_chain = LLMChain(llm=llm, prompt=get_summary_prompt())
    # Use StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain

app = typer.Typer(add_completion=False)

@app.command()
def main(model: str = typer.Option("mixtral-8x7b-instruct", help="The LLM model to use"),
          key: Optional[str] = typer.Option(None, help="The API key for the LLM model"),
          interactive: bool = typer.Option(False, "--interactive", "-i", help="If true enter interactive mode"),
          url: str = typer.Argument(help="The URL of the video to summarize")):
    """
    This is a YouTube Summarizer and Q&A CLI using Perplexity AI and ChatGPT-4. Supply your API key and the desired model (or set them as ${OPENAI_API_KEY} and ${PPLX_API_KEY}) to start summarizing and asking questions about the video in an interactive prompt. Type "quit" to exit the Q&A.

    Note: To use GPT-4 (Turbo), use the model name "gpt-4" and supply the OpenAI API key.
    """
    # check if the key is provided, if not abort
    if model=="gpt-4":
        model="gpt-4-turbo-preview"
        key = key or os.getenv("OPENAI_API_KEY")
    else:
        key = key or os.getenv("PPLX_API_KEY")
    if key is None:
        typer.echo(typer.style("Error: An API key is required.", fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)

    
    # get youtube transcript 
    try: 
        docs = get_youtube_transcript(url)
    except ValueError:
        typer.echo(typer.style("Error: Invalid YouTube URL.", fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)

    # get the llm 
    llm = get_llm(model, key)
    # summarize & print result
    try:
        response = get_summary_chain(llm).invoke(docs)
        typer.echo(response['output_text'])
    except openai.AuthenticationError:
        typer.echo(typer.style(f"Error: Invalid API key.", fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)
    except openai.BadRequestError as e:
        typer.echo(typer.style(f'There was an error querying the LLM: {e.message}', fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)
    except:
        typer.echo(typer.style(f"An error occurred.", fg=typer.colors.RED, bold=True), err=True)
        raise typer.Exit(code=1)


    if not interactive:
        return
    import validators
    from urllib.parse import urlparse
    #Enter interactive chat session if indicated
    chat = create_stuff_documents_chain(llm, get_qa_prompt())
    history = ChatMessageHistory()
    while True:
        user_input = prompt("\n>> ")
        if user_input.lower() == "quit" or user_input.lower() == "exit":
            break
        elif user_input == "":
            continue
        elif validators.url(user_input): #if user enters another url
            pass
        
        history.add_user_message(user_input)
        response = chat.invoke({
            "messages": history.messages,
            "context": docs
            })
        typer.echo(response)
        history.add_ai_message(response)
    typer.echo("Exiting")


if __name__ == "__main__":
    app()


