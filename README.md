## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Research information is often distributed across multiple documents, making it difficult for users to quickly locate relevant details. Manually searching through several research articles can be time-consuming and inefficient. To overcome this challenge, a multidocument retrieval agent can be developed that automatically processes multiple documents and retrieves the most relevant information based on user queries.

By using LlamaIndex, the system can index multiple documents and enable efficient search and retrieval. The agent analyzes the indexed information and generates concise and accurate responses to user questions.

### DESIGN STEPS:

### STEP 1:
Load and preprocess multiple research articles using LlamaIndex document loaders and convert them into structured nodes for indexing.

### STEP 2:
Create a document index and configure the retrieval mechanism to search across all indexed documents for relevant information.

### STEP 3:
Integrate the retrieval system with a language model to process user queries and generate meaningful responses based on the retrieved content.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=CCSPm6V5EF",
    "https://openreview.net/pdf?id=MS9nWFY7LG",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Summarize the document DOC1, "
    "and also DOC2"
)
response = agent.query("What are the similarities in DOC1,DOC2 and DOC3")
print(str(response))
```
### OUTPUT:


### RESULT:
The multidocument retrieval agent was successfully implemented using LlamaIndex. The system efficiently retrieved relevant information from multiple documents and provided concise, accurate answers to user queries.
