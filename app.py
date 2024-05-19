#!/home/lilly/code/video-sum/bin/python3


import openai
from straico import ChatStraico
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
from urllib.parse import urlparse
from dotenv import load_dotenv

def get_youtube_transcript(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    return docs


def get_summary_prompt(short):
    if short:
        prompt_template = """Write a concise summary of the following, highlighting the key points.
            <{text}>"""
    else:
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
    if model[0:3] == "gpt":
        llm = ChatOpenAI(temperature=0, model=model, openai_api_key=key)
    else:
        # llm = ChatPerplexity(temperature=0, model=model, pplx_api_key=key)
        llm = ChatStraico(model=model, straico_api_key=key)
    return llm


def get_summary_chain(llm, short):
    llm_chain = LLMChain(llm=llm, prompt=get_summary_prompt(short))
    # Use StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )
    return stuff_chain


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option("mixtral-8x7b-instruct", help="The LLM model to use"),
    key: Optional[str] = typer.Option(None, help="The API key for the LLM model"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="If true, enter interactive mode"
    ),
    short: bool = typer.Option(
        False, "--short", "-s", help="If true, returns a concise summary"
    ),
    url: str = typer.Argument(
        None, help="The URL of the video to summarize (Optional)"
    ),
):
    """
    This is a YouTube Summarizer and Q&A CLI using Perplexity AI and ChatGPT-4. Supply your API key and the desired model or set the keys as ${OPENAI_API_KEY} and ${PPLX_API_KEY}.

    Start summarizing and asking questions about the video in an interactive prompt. You can also enter a new YouTube video URL in the interactive prompt and it will return the summary and start the Q&A session for this video. Type "quit" to exit the Q&A.

    Note: To use GPT-4 (Turbo), use the model name "gpt-4" and supply the OpenAI API key.
    """
    # check if the key is provided, if not abort

    load_dotenv()
    model = model or os.getenv("MODEL")
    if model == "gpt-4":
        model = "gpt-4-turbo-preview"
        key = key or os.getenv("OPENAI_API_KEY")
    elif model == "gpt-3":
        model = "gpt-3.5-turbo"
        key = key or os.getenv("OPENAI_API_KEY")
    else:
        key = key or os.getenv("STRAICO_API_KEY")
    if key is None:
        typer.echo(
            typer.style(
                "Error: An API key is required.", fg=typer.colors.RED, bold=True
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    # get the llm
    llm = get_llm(model, key)

    empty_url = True

    # loop over urls
    while True:
        if url:
            empty_url = False
            typer.echo("Summarizing...")
            # get youtube transcript
            try:
                docs = get_youtube_transcript(url)
            except ValueError:
                typer.echo(
                    typer.style(
                        "Error: Invalid YouTube URL.", fg=typer.colors.RED, bold=True
                    ),
                    err=True,
                )
                raise typer.Exit(code=1)
            # summarize & print result
            try:
                response = get_summary_chain(llm, short).invoke(docs)
                typer.echo(response["output_text"])
                typer.echo("")
            except openai.AuthenticationError:
                typer.echo(
                    typer.style(
                        f"Error: Invalid API key.", fg=typer.colors.RED, bold=True
                    ),
                    err=True,
                )
                raise typer.Exit(code=1)
            except openai.BadRequestError as e:
                typer.echo(
                    typer.style(
                        f"There was an error querying the LLM: {e.message}",
                        fg=typer.colors.RED,
                        bold=True,
                    ),
                    err=True,
                )
                raise typer.Exit(code=1)
            except:
                typer.echo(
                    typer.style(f"An error occurred.", fg=typer.colors.RED, bold=True),
                    err=True,
                )
                raise typer.Exit(code=1)

        if not interactive:
            return
        # Enter interactive chat session if indicated
        chat = create_stuff_documents_chain(llm, get_qa_prompt())
        history = ChatMessageHistory()
        while True:
            user_input = prompt(">> ")
            if user_input.lower() == "quit" or user_input.lower() == "exit":
                typer.echo("Exiting")
                raise typer.Exit()
            # if user enters another/new url
            elif urlparse(user_input).scheme:
                url = user_input
                break
            elif empty_url:
                typer.echo("Enter a YouTube URL to begin asking questions.")
                continue
            elif user_input == "":
                typer.echo("Enter a question.")
                continue

            history.add_user_message(user_input)
            response = chat.invoke({"messages": history.messages, "context": docs})
            typer.echo(response)
            typer.echo("")
            history.add_ai_message(response)


if __name__ == "__main__":
    app()
