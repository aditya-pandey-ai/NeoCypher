import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

# Page configuration
st.set_page_config(page_title="CypherQuery: Chat with your DB")
st.title("CypherQuery: Chat with DB")

# Database options
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
NEO = "USE_NEO"

# Sidebar options
radio_opt = [
    "Use SQLite 3 Database - test.db",
    "Connect to your SQL database",
    "Connect to your Neo4j database"
]

selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL user")
    mysql_password = st.sidebar.text_input("MySQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
elif radio_opt.index(selected_opt) == 2:
    db_uri = NEO
else:
    db_uri = LOCALDB

# Groq API Key input
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not db_uri:
    st.info("Please provide a MySQL database URI and information")
if not api_key:
    st.info("Please provide a Groq API Key")

# LLM model setup
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)


# Configure MySQL or SQLite database
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri:
        if db_uri == LOCALDB:
            dbfilepath = (Path(__file__).parent / "test.db").absolute()
            creator = lambda: sqlite3.connect(f"file://{dbfilepath}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        elif db_uri == MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            return SQLDatabase(
                create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))


# Configure Neo4j database
def configure_neo4j():
    NEO4J_URI = st.sidebar.text_input("Neo4j URI")
    NEO4J_USERNAME = st.sidebar.text_input("Neo4j Username")
    NEO4J_PASSWORD = st.sidebar.text_input("Neo4j Password", type="password")

    if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
        try:
            graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
            return graph
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")
            return None
    else:
        st.info("Please provide Neo4j connection details.")
        return None


# Database connection handling
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
elif db_uri == NEO:
    db = configure_neo4j()
else:
    db = configure_db(db_uri)

# Toolkit setup for SQL databases
if db_uri != NEO and db:
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

# Neo4j specific handling with GraphCypherQAChain using invoke
if db_uri == NEO and db:
    chain = GraphCypherQAChain.from_llm(graph=db, llm=llm, verbose=True)

# Session state to manage chat history
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Displaying chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User query input and response handling
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())

        if db_uri == NEO and db:
            response = chain.invoke({"query": user_query})
        else:
            response = agent.run(user_query, callbacks=[streamlit_callback])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
