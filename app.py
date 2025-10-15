# file: app.py

import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from tools import create_work_order

# --- App Configuration ---
st.set_page_config(page_title="Kansas DOT AI Assistant", page_icon=" Kansas DOT AI Assistant")
st.title("AI Assistant")
st.caption("Your smart assistant for road information and maintenance reporting.")

# --- Core Logic Caching ---
@st.cache_resource
def get_chatbot_components():
    """
    Initializes all the necessary components for the chatbot, including memory.
    """
    print("--- Initializing chatbot components (with memory)... ---")

    # --- Setup ---
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    if "MONGO_URI" not in os.environ:
        raise ValueError("MONGO_URI not found. Please set it in your .env file.")

    agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)
    framing_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, convert_system_message_to_human=True)

    # --- 1. Define Tools ---
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
    tools = [create_work_order, retriever_tool]

    # --- 2. Create the Agent with Memory ---
    agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant for a state's Department of Transportation.
    Your primary goal is to be a **diligent and thorough information gatherer** when a user wants to file a maintenance report.

    When a user reports an issue, your job is to guide them to provide all the necessary details for the 'create_work_order' tool.
    - You MUST get the `location`, `issue`, and `priority`.
    - You SHOULD ALWAYS proactively ask for the optional but highly useful details: `direction` of travel and a nearby `landmark`. Do not assume you don't need them. A more detailed report is always better.
    - Only after gathering all these details should you call the 'create_work_order' tool.

    If the user is just asking a general question, use the 'transportation_qa_search' tool."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(agent_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False
    )

    # --- 3. Create the Answer-Framing Chain ---
    framing_prompt = ChatPromptTemplate.from_template(
        """You are a friendly and helpful chatbot spokesperson for the state DOT.
        Your job is to take the raw information or tool output and frame it as a clear, helpful, and conversational answer.

        Original Question: {question}
        Tool Output / Raw Information: {context}

        Your Conversational Answer:"""
    )
    framing_chain = (
        RunnablePassthrough()
        | framing_prompt
        | framing_llm
        | StrOutputParser()
    )

    return agent_executor, framing_chain

# --- Main Application Logic ---

agent_executor, framing_chain = get_chatbot_components()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today with Kansas transportation?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about road conditions or report an issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Prepare the chat history for the agent
            history = st.session_state.messages[:-1]
            chat_history_for_agent = []
            for msg in history:
                if msg["role"] == "user":
                    chat_history_for_agent.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history_for_agent.append(AIMessage(content=msg["content"]))

            # Invoke the agent with the new input and the prepared chat history
            agent_result = agent_executor.invoke({
                "input": prompt,
                "chat_history": chat_history_for_agent
            })
            raw_output = agent_result["output"]

            # Use the framing chain to polish the agent's raw output
            final_answer = framing_chain.invoke({
                "question": prompt,
                "context": raw_output
            })
            
            st.markdown(final_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": final_answer})