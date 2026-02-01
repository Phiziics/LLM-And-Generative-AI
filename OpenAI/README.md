# OpenAI Experiments and Labs

This folder contains hands on notebooks and helper code demonstrating how to work with the OpenAI API, from basic setup through prompting, tokenization, and building a small real world business style solution.

The material is structured as a learning progression. Each notebook builds on the previous one.

---

## Learning Objectives

By completing these notebooks you will learn how to

• Set up and validate OpenAI API access  
• Call OpenAI endpoints programmatically  
• Understand and measure token usage  
• Design effective prompts for business use cases  
• Combine web scraping with LLMs to create practical products  

---

## Folder Contents

### 1.openai_intro.ipynb

**Purpose**  
Introduction to OpenAI usage through a simple real world task.

**What it does**  
• Loads environment variables  
• Uses a web scraper to extract website text  
• Sends extracted content to an OpenAI model  
• Displays structured results inside the notebook  

**Key concepts**  
• API client initialization  
• Environment variable management  
• Passing external data into LLM prompts  

**When to use**  
Start here if you are new to OpenAI or want a clean reference for basic setup and first calls.

---

### 2.openai_endpoint.ipynb

**Purpose**  
Understand how OpenAI endpoints work under the hood.

**What it does**  
• Validates API key format and availability  
• Demonstrates making requests using the requests library  
• Shows how to construct raw HTTP calls to OpenAI endpoints  

**Key concepts**  
• Endpoint structure  
• Request payloads  
• Error handling and validation  

**When to use**  
Useful if you want to understand what the OpenAI SDK is abstracting away or if you plan to integrate OpenAI into non Python environments.

---

### 3.tokenizing.ipynb

**Purpose**  
Explain and demonstrate tokenization.

**What it does**  
• Introduces tokenization and why it matters  
• Uses `tiktoken` to tokenize text  
• Counts tokens for different inputs  
• Shows how token count affects cost and limits  

**Key concepts**  
• Tokens vs characters  
• Prompt size management  
• Cost awareness  

**When to use**  
Critical reference notebook when designing production prompts or working with large inputs.

---

### 4.openai_prompting.ipynb

**Purpose**  
Build a complete prompting based business solution.

**What it does**  
• Accepts a company name and website  
• Scrapes site content and links  
• Designs structured prompts  
• Generates a professional business brochure using an LLM  

**Key concepts**  
• Prompt engineering  
• Multi step prompting workflows  
• Business focused LLM outputs  
• Using LLMs as product components  

**When to use**  
This is the most advanced notebook and represents how LLMs are used in real products.

---

### web_scraper.py

**Purpose**  
Reusable helper utilities for extracting website content.

**What it does**  
• Fetches website text content  
• Removes irrelevant elements like scripts and styles  
• Truncates content for safe LLM usage  
• Extracts all links from a webpage  

**Functions included**  
• `fetch_website_contents(url)`  
• `fetch_website_links(url)`  

This file is imported and used across multiple notebooks.

:contentReference[oaicite:0]{index=0}

---

## Recommended Execution Order

1. `1.openai_intro.ipynb`  
2. `2.openai_endpoint.ipynb`  
3. `3.tokenizing.ipynb`  
4. `4.openai_prompting.ipynb`  

Running them in order ensures concepts build correctly.

---

## Setup Instructions

### 1. Create a virtual environment

```python
# Mac or Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
py -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install openai python-dotenv requests beautifulsoup4 tiktoken
Set environment variables

Create a .env file in this folder:

OPENAI_API_KEY="your_api_key_here"

Never commit .env to source control.