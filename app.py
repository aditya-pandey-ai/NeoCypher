import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="CypherQuery: Chat with SQL DB")
st.title("CypherQuery: Chat with SQL DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"
MONGO="USE_MONGO"
NEO = "USE_NEO"

radio_opt = ["Use SQLLite 3 Database- test.db","Connect to your SQL database","Connect to your MongoDB database","Connect to your NEO database"]

selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options = radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri=MYSQL
    mysql_host = st.sidebar.text_input("Provide MY SQL Host")
    mysql_user = st.sidebar.text_input("MYSQL user")
    mysql_password = st.sidebar.text_input("MYSQL password",type="password")
    mysql_db = st.sidebar.text_input("MYSQL database")

elif radio_opt.index(selected_opt) == 2:
    db_uri=MONGO

else:
    db_uri = LOCALDB

api_key=st.sidebar.text_input(label="Groq API Key",type ="password")

if not db_uri:
    st.info("Please provide an MYSQL database URI and information")

if not api_key:
    st.info("Please provide an Groq API Key")

## LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192",streaming = True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri:
        if db_uri==LOCALDB:
            dbfilepath=(Path(__file__).parent/"test.db").absolute()
            print(dbfilepath)
            creator = lambda: sqlite3.connect(f"file://{dbfilepath}?mode=ro",uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        elif db_uri==MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all Mysql Connection details.")
                st.stop()
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
if db_uri == MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db,llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
