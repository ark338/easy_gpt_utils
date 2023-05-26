![PyPI](https://img.shields.io/pypi/v/easy_gpt_utils?color=g)

# easy_gpt_utils
Easy GPT Utils include 1. chat completion 2. embedding and 3. vector database and others to help create app based on GPT and vector search
(vector database is not implemented yet)

## gpt
gpt is a simple util to use OpenAI's chat modles such as gpt-3.5-terbo gpt-4 etc. It has functions:
1. calculate token of a string
2. split string by paragraph and sentences within maximum token limit
3. set model, temprature
4. set system prompt and post prompt
5. add context infomation to query
6. remember talking history(buggy, will exceed token limit)

## embedding
embedding is a simple util to create embedding (1536 dimension victor) for strings and paragraphs, It has functions:
1. create embedding from string ,paragrahp and file
2. save embeddings to file

## vector_database
support pinecorn as vector database now


## latest version
V0.1.10


### In summary, easy_gpt_utils is now on early stage and hope it could help to create more powerful AI Apps in the future
