# youtube-summarizer-qa
YouTube Video Summarizer and Q&A CLI using Perplexity AI &amp; ChatGPT, built with langchain.

Supply your API key (```--api```) and the desired model (```--model```). Alternatively you can set the API keys as environmental variables ```OPENAI_API_KEY``` and ```PPLX_API_KEY```. It supports all [Perplexity Models](https://docs.perplexity.ai/docs/model-cards) and ChatGPT-4-Turbo (```--model=gpt-4```) & ChatGPT-3.5. Default model is Mixtral-8x7B Instruct, which works the best for me. 

The program will return a summary of the video covering each section (including timestamps), and if it was invoked with the ```--interactive``` flag, enters an interactive session where you can type questions about the video. You can also enter a new YouTube video URL in the interactive prompt and it will summarize it and start the Q&A session for this video. Type "quit" to exit the Q&A. 

There is also a ```--short``` mode that provides a concise summary instead of one covering every section of the video. 
