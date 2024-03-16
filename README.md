# youtube-summarizer-qa
YouTube Video Summarizer and Q&A CLI using Perplexity AI &amp; ChatGPT-4, built with langchain.

Supply your API key (```--api```) and the desired model (```--model```) to start summarizing & asking questions about the video. It supports all [Perplexity Models](https://docs.perplexity.ai/docs/model-cards) and ChatGPT-4-Turbo. Default model is Mixtral-8x7B Instruct, which works the best for me. 

After summarizing the video, if provided with the ```--interactive``` flag on invocation, you enter an interactive session where you can type questions about the video. 

There is also a ```--short``` mode that provides a concise summary instead of one covering every section of the video. 
