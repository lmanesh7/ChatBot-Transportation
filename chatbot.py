# file: chatbot.py (Updated with two Google models)

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from tools import create_work_order

# --- Setup ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# 🧠 AGENT LLM (The "Dispatcher"): High-quality model for reasoning
agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)

# 🗣️ FRAMING LLM (The "Spokesperson"): Fast model for polishing answers
framing_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, convert_system_message_to_human=True)

# --- 1. Define Tools ---
tools = [create_work_order]

# Setup the retriever tool for Q&A
client = MongoClient(os.environ["MONGO_URI"])
db = client.transportationDB
collection = db.knowledge_base
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    index_name="vector_index_knowledge"
)
retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "transportation_qa_search",
    "Use this tool to search for information and answer user questions about state transportation, road conditions, and policies."
)
tools.append(retriever_tool)

# --- 2. Create the Agent ---
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for a state's Department of Transportation. You have two primary functions: 1. Answer general questions. 2. Create work orders for maintenance reports. First, understand the user's intent. If they are asking a question, use 'transportation_qa_search'. If they are reporting an issue, use 'create_work_order'. Output only the direct result from the tool call."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(agent_llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 3. Create the Answer-Framing Chain ---
framing_prompt = ChatPromptTemplate.from_template(
    """You are a friendly and helpful chatbot spokesperson for the state DOT.
    Your job is to take the raw information or tool output and frame it as a clear, helpful, and conversational answer.

    Original Question: {question}
    Tool Output / Raw Information: {context}

    Your Conversational Answer:"""
)

# This chain will feed the raw output from the agent into the framing LLM
framing_chain = (
    RunnablePassthrough()
    | framing_prompt
    | framing_llm
    | StrOutputParser()
)


# --- 4. Run the Chatbot ---
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break
        
        # First, the agent runs to get the raw, factual answer
        agent_result = agent_executor.invoke({"input": user_input})
        raw_output = agent_result["output"]

        # Then, the framing chain polishes the raw output into a nice response
        final_answer = framing_chain.invoke({
            "question": user_input,
            "context": raw_output
        })
        
        print("Bot:", final_answer)