# youtube-summarizer-qa
YouTube Video Summarizer and Q&A CLI using Perplexity AI &amp; ChatGPT, built with langchain.

Supply your API key (```--api```) and the desired model (```--model```). Alternatively you can set the API keys as environmental variables ```OPENAI_API_KEY``` and ```PPLX_API_KEY```. It supports all [Perplexity Models](https://docs.perplexity.ai/docs/model-cards) and ChatGPT-4-Turbo (```--model=gpt-4```) & ChatGPT-3.5. Default model is Mixtral-8x7B Instruct, which works the best for me. 

The program will return a summary of the video covering each section (including timestamps), and if it was invoked with the ```--interactive``` flag, enters an interactive session where you can type questions about the video. You can also enter a new YouTube video URL in the interactive prompt and it will summarize it and start the Q&A session for this video. Type "quit" to exit the Q&A. 

There is also a ```--short``` mode that provides a concise summary instead of one covering every section of the video. 

## How to run it 

Create a python virtual environment and activate it 
```console
python3 -m venv myenv
source myenv/bin/activate
```

Inside the venv install the packages. 
```console
pip3 install -r requirements.txt
```

Now you can run the app
```console
$ python3 app.py --help

Usage: app.py [OPTIONS] [URL]

  This is a YouTube Summarizer and Q&A CLI using Perplexity AI and ChatGPT-4.
  Supply your API key and the desired model or set the keys as
  ${OPENAI_API_KEY} and ${PPLX_API_KEY}.

  Start summarizing and asking questions about the video in an interactive
  prompt. You can also enter a new YouTube video URL in the interactive prompt
  and it will return the summary and start the Q&A session for this video.
  Type "quit" to exit the Q&A.

  Note: To use GPT-4 (Turbo), use the model name "gpt-4" and supply the OpenAI
  API key.

Arguments:
  [URL]  The URL of the video to summarize (Optional)

Options:
  --model TEXT       The LLM model to use  [default: mixtral-8x7b-instruct]
  --key TEXT         The API key for the LLM model
  -i, --interactive  If true, enter interactive mode
  -s, --short        If true, returns a concise summary
  --help             Show this message and exit.

```
