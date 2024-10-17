import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
from pymongo import MongoClient
import sqlite3
from pathlib import Path

# Constants
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
MONGODB = "USE_MONGODB"

# Streamlit page config
st.set_page_config(page_title="CypherQuery: Chat with DB")
st.title("CypherQuery: Chat with DB")

# Sidebar options
db_options = ["Use SQLite 3 Database - test.db", "Connect to your SQL database", "Use MongoDB"]
selected_db_option = st.sidebar.radio("Choose the DB you want to chat with", db_options)

# Initialize db_uri
db_uri = None

# MySQL connection inputs
if selected_db_option == "Connect to your SQL database":
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
# MongoDB connection inputs
elif selected_db_option == "Use MongoDB":
    db_uri = MONGODB
    mongo_uri = st.sidebar.text_input("Enter MongoDB URI")
    mongo_db_name = st.sidebar.text_input("Enter Database Name")
    mongo_collection_name = st.sidebar.text_input("Enter Collection Name")
# SQLite inputs
else:
    db_uri = LOCALDB

# Groq API key input
api_key = st.sidebar.text_input("Groq API Key", type="password")

# Validation of inputs
if not db_uri:
    st.info("Please provide database information.")
if not api_key:
    st.info("Please provide a Groq API Key.")


# Initialize LLM using Groq
def get_llm():
    return ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)


llm = get_llm()


# Function to configure SQL database (SQLite or MySQL)
@st.cache_resource(ttl="2h")
def configure_sql_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_file = (Path(__file__).parent / "test.db").absolute()
        creator = lambda: sqlite3.connect(f"file://{db_file}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL and mysql_host and mysql_user and mysql_password and mysql_db:
        return SQLDatabase(
            create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    else:
        return None  # Return None if inputs are incomplete or invalid


# Function to configure MongoDB
def configure_mongo_db(mongo_uri, mongo_db_name, mongo_collection_name):
    client = MongoClient(mongo_uri)
    db = client[mongo_db_name]
    collection = db[mongo_collection_name]
    return client, collection


# Function to get the schema of a MongoDB collection dynamically
def get_collection_schema(collection):
    sample_docs = collection.find().limit(10)
    schema = {}

    for doc in sample_docs:
        for field in doc:
            field_type = type(doc[field]).__name__
            if field not in schema:
                schema[field] = field_type
            elif schema[field] != field_type:
                schema[field] = "mixed"
    return schema


# Dynamic MongoDB query conversion based on natural language input
def convert_nl_to_mongo_query(nl_query, collection_schema):
    # Directly handle requests for all documents in the collection
    if "give me all the data" in nl_query.lower() or "fetch all" in nl_query.lower():
        return {}  # Empty query fetches all documents

    # Define a prompt template for other types of queries
    template = """
    You are a MongoDB query generator. Based on the given collection schema, generate a valid MongoDB query from the following natural language query:

    Schema: {schema}

    Query: {user_query}

    MongoDB query:
    """

    # Populate schema and query
    prompt = PromptTemplate(input_variables=["schema", "user_query"], template=template)
    filled_prompt = prompt.format(schema=collection_schema, user_query=nl_query)

    # Use the LLM to process the prompt and generate the query
    generated_query = llm.generate(filled_prompt)

    # Convert the generated string into an actual dictionary (MongoDB query)
    mongo_query = eval(generated_query)

    return mongo_query

# SQL/MongoDB Toolkit Initialization
tools = []
db = None

if db_uri == MYSQL:
    db = configure_sql_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    if db is not None:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
    else:
        st.error("MySQL connection parameters are incomplete or invalid.")
elif db_uri == LOCALDB:
    db = configure_sql_db(db_uri)
    if db is not None:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
    else:
        st.error("SQLite configuration failed.")
elif db_uri == MONGODB and mongo_uri and mongo_db_name and mongo_collection_name:
    client, collection = configure_mongo_db(mongo_uri, mongo_db_name, mongo_collection_name)
    # Fetch MongoDB schema
    collection_schema = get_collection_schema(collection)


    # Tool for MongoDB queries
    def mongo_tool(query):
        mongo_query = convert_nl_to_mongo_query(query, collection_schema)
        results = collection.find(mongo_query)
        return list(results)


    # Create Tool object for MongoDB
    mongo_tool = Tool(
        name="MongoDBQuery",
        func=mongo_tool,
        description="Query the MongoDB database."
    )
    tools = [mongo_tool]  # Use MongoDB tool directly

# Initialize the agent for both SQL and MongoDB
if tools:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# Chat history and user input
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    if db_uri == MONGODB:
        # MongoDB query handling
        response = mongo_tool(user_query)
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        st.write(response)
    else:
        # SQL query handling
        if agent:
            with st.chat_message("assistant"):
                response = agent.run(user_query)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
        else:
            st.error("No database tools available for querying.")
