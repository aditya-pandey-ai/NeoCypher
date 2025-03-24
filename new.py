from typing import List, Dict, Any, Optional, Union
from langchain_groq import ChatGroq
from langchain.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import os
import asyncio
from functools import wraps
import pymongo
import psycopg2
from pymongo import MongoClient
from psycopg2.extras import RealDictCursor
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

def async_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class DatabaseConnector:
    """Base class for database connections"""

    def __init__(self):
        self.connection = None

    def connect(self):
        """Establish connection to the database"""
        raise NotImplementedError

    def disconnect(self):
        """Close the database connection"""
        raise NotImplementedError

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame"""
        raise NotImplementedError

    def get_schema_info(self) -> str:
        """Get database schema information"""
        raise NotImplementedError


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector"""

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.db = None

    def connect(self):
        """Connect to SQLite database"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        try:
            self.connection = sqlite3.connect(self.db_path)
            # Verify connection works
            self.connection.cursor().execute("SELECT 1")
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            return True
        except Exception as e:
            raise Exception(f"Failed to initialize SQLite connection: {str(e)}")

    def disconnect(self):
        """Close the SQLite connection"""
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        if not query:
            raise ValueError("SQL query cannot be empty")

        try:
            # Split into individual statements
            statements = self._split_sql_statements(query)
            if not statements:
                raise ValueError("No valid SQL statements found")

            try:
                # Start a transaction
                self.connection.execute("BEGIN TRANSACTION")

                # Execute all non-SELECT statements first
                cursor = self.connection.cursor()
                for stmt in statements[:-1]:  # All but the last statement
                    if stmt.strip().upper().startswith('SELECT'):
                        continue  # Skip SELECT statements except the last one
                    cursor.execute(stmt)

                # Commit the transaction
                self.connection.commit()

                # Execute the final statement (should be a SELECT) and return results
                final_stmt = statements[-1]
                if not final_stmt.strip().upper().startswith('SELECT'):
                    # If last statement isn't a SELECT, execute it and return empty DataFrame
                    cursor.execute(final_stmt)
                    self.connection.commit()
                    return pd.DataFrame()

                # Return results of the final SELECT statement
                df = pd.read_sql_query(final_stmt, self.connection)
                return df if not df.empty else pd.DataFrame()

            except Exception as e:
                self.connection.rollback()
                raise
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

    def _split_sql_statements(self, query: str) -> List[str]:
        """Split multiple SQL statements into a list of individual statements."""
        # Split on semicolon but ignore semicolons inside quotes
        statements = []
        current_statement = []
        in_quotes = False
        quote_char = None

        for char in query:
            if char in ["'", '"'] and (not quote_char or char == quote_char):
                in_quotes = not in_quotes
                quote_char = char if in_quotes else None

            if char == ';' and not in_quotes:
                current_statement = ''.join(current_statement).strip()
                if current_statement:
                    statements.append(current_statement)
                current_statement = []
            else:
                current_statement.append(char)

        # Add the last statement if it exists
        final_statement = ''.join(current_statement).strip()
        if final_statement:
            statements.append(final_statement)

        return [stmt for stmt in statements if stmt]

    def get_schema_info(self) -> str:
        """Get SQLite schema information"""
        return self.db.get_table_info()

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about database tables and their schemas."""
        tables = {}
        try:
            cursor = self.connection.cursor()

            cursor.execute("""
                   SELECT name FROM sqlite_master 
                   WHERE type='table' AND name NOT LIKE 'sqlite_%';
               """)

            table_names = cursor.fetchall()
            if not table_names:
                raise ValueError("No tables found in database")

            for (table_name,) in table_names:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                if not columns:
                    continue

                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_data = cursor.fetchall()

                tables[table_name] = {
                    'columns': [col[1] for col in columns],
                    'sample_data': sample_data if sample_data else []
                }

            return tables
        except Exception as e:
            raise Exception(f"Error getting table info: {str(e)}")


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector for Neon DB"""

    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.db = None

    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            # Verify connection works
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

            self.db = SQLDatabase.from_uri(self.connection_string)
            return True
        except Exception as e:
            raise Exception(f"Failed to initialize PostgreSQL connection: {str(e)}")

    def disconnect(self):
        """Close the PostgreSQL connection"""
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute PostgreSQL query and return results as DataFrame"""
        if not query:
            raise ValueError("SQL query cannot be empty")

        try:
            # Split into individual statements
            statements = self._split_sql_statements(query)
            if not statements:
                raise ValueError("No valid SQL statements found")

            try:
                # Start a transaction
                self.connection.autocommit = False

                # Execute all non-SELECT statements first
                cursor = self.connection.cursor()
                for stmt in statements[:-1]:  # All but the last statement
                    if stmt.strip().upper().startswith('SELECT'):
                        continue  # Skip SELECT statements except the last one
                    cursor.execute(stmt)

                # Execute the final statement and return results
                final_stmt = statements[-1]
                if not final_stmt.strip().upper().startswith('SELECT'):
                    # If last statement isn't a SELECT, execute it and return empty DataFrame
                    cursor.execute(final_stmt)
                    self.connection.commit()
                    return pd.DataFrame()

                # Return results of the final SELECT statement
                cursor = self.connection.cursor(cursor_factory=RealDictCursor)
                cursor.execute(final_stmt)
                results = cursor.fetchall()
                self.connection.commit()

                # Convert to DataFrame
                if results:
                    df = pd.DataFrame(results)
                    return df
                return pd.DataFrame()

            except Exception as e:
                self.connection.rollback()
                raise
        except Exception as e:
            raise Exception(f"Error executing PostgreSQL query: {str(e)}")

    def _split_sql_statements(self, query: str) -> List[str]:
        """Split multiple SQL statements into a list of individual statements."""
        # Split on semicolon but ignore semicolons inside quotes
        statements = []
        current_statement = []
        in_quotes = False
        quote_char = None

        for char in query:
            if char in ["'", '"'] and (not quote_char or char == quote_char):
                in_quotes = not in_quotes
                quote_char = char if in_quotes else None

            if char == ';' and not in_quotes:
                current_statement = ''.join(current_statement).strip()
                if current_statement:
                    statements.append(current_statement)
                current_statement = []
            else:
                current_statement.append(char)

        # Add the last statement if it exists
        final_statement = ''.join(current_statement).strip()
        if final_statement:
            statements.append(final_statement)

        return [stmt for stmt in statements if stmt]

    def get_schema_info(self) -> str:
        """Get PostgreSQL schema information"""
        return self.db.get_table_info()

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about PostgreSQL tables and their schemas."""
        tables = {}
        try:
            cursor = self.connection.cursor()

            # Get table names
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)

            table_names = cursor.fetchall()
            if not table_names:
                raise ValueError("No tables found in database")

            for (table_name,) in table_names:
                # Get column information
                cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}';
                """)

                columns = cursor.fetchall()
                if not columns:
                    continue

                # Get sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_data = cursor.fetchall()

                tables[table_name] = {
                    'columns': [col[0] for col in columns],
                    'sample_data': sample_data if sample_data else []
                }

            return tables
        except Exception as e:
            raise Exception(f"Error getting PostgreSQL table info: {str(e)}")


class MongoDBConnector(DatabaseConnector):
    """MongoDB Atlas connector"""

    def __init__(self, connection_string: str, database_name: str):
        super().__init__()
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None

    def connect(self):
        """Connect to MongoDB database"""
        try:
            # Import certifi for TLS certificate verification
            import certifi

            # Connect with TLS certificate verification
            self.client = MongoClient(
                self.connection_string,
                tlsCAFile=certifi.where(),
                server_api=ServerApi('1')  # Using the stable API version
            )

            # Verify connection works by pinging the deployment
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB Atlas!")

            self.db = self.client[self.database_name]
            return True
        except Exception as e:
            raise Exception(f"Failed to initialize MongoDB connection: {str(e)}")

    def disconnect(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute MongoDB query and return results as DataFrame"""
        if not query:
            raise ValueError("MongoDB query cannot be empty")

        try:
            # Execute MongoDB query
            # The query is expected to be a Python dictionary in string format
            query_dict = eval(query)

            # Validate query structure
            if not isinstance(query_dict, dict) or 'collection' not in query_dict or 'operation' not in query_dict:
                raise ValueError(
                    "Invalid MongoDB query format. Expected dictionary with 'collection' and 'operation' keys.")

            collection_name = query_dict['collection']
            operation = query_dict['operation']
            collection = self.db[collection_name]

            # Handle different CRUD operations
            if operation == 'find':
                filter_query = query_dict.get('filter', {})
                projection = query_dict.get('projection', None)
                limit = query_dict.get('limit', 0)

                if projection:
                    cursor = collection.find(filter_query, projection).limit(limit)
                else:
                    cursor = collection.find(filter_query).limit(limit)

                results = list(cursor)

            elif operation == 'aggregate':
                pipeline = query_dict.get('pipeline', [])
                cursor = collection.aggregate(pipeline)
                results = list(cursor)

            elif operation == 'update':
                filter_query = query_dict.get('filter', {})
                update_doc = query_dict.get('update', {})
                many = query_dict.get('many', False)

                if many:
                    result = collection.update_many(filter_query, update_doc)
                else:
                    result = collection.update_one(filter_query, update_doc)

                # Return update statistics as a DataFrame
                results = [{
                    'matched_count': result.matched_count,
                    'modified_count': result.modified_count
                }]

            elif operation == 'insert':
                documents = query_dict.get('documents', [])

                if not documents:
                    raise ValueError("No documents provided for insert operation")

                # Perform insert many or single insert based on document list
                if len(documents) > 1:
                    result = collection.insert_many(documents)
                    results = [{'inserted_ids': list(map(str, result.inserted_ids))}]
                else:
                    result = collection.insert_one(documents[0])
                    results = [{'inserted_id': str(result.inserted_id)}]

            elif operation == 'delete':
                filter_query = query_dict.get('filter', {})
                many = query_dict.get('many', False)

                if many:
                    result = collection.delete_many(filter_query)
                else:
                    result = collection.delete_one(filter_query)

                # Return delete statistics as a DataFrame
                results = [{
                    'deleted_count': result.deleted_count
                }]

            else:
                raise ValueError(f"Unsupported MongoDB operation: {operation}")

            # Convert to DataFrame
            if results:
                df = pd.DataFrame(results)

                # Drop MongoDB _id column if it exists and is not needed
                if '_id' in df.columns and not query_dict.get('include_id', False):
                    df = df.drop('_id', axis=1)

                return df

            return pd.DataFrame()

        except Exception as e:
            raise Exception(f"Error executing MongoDB query: {str(e)}")

    def get_schema_info(self) -> str:
        """Get MongoDB schema information"""
        schema_info = "MongoDB Collections:\n"

        for collection_name in self.db.list_collection_names():
            schema_info += f"- Collection: {collection_name}\n"

            # Sample a document to infer schema
            sample = self.db[collection_name].find_one()
            if sample:
                schema_info += "  Fields:\n"
                for field, value in sample.items():
                    if field == "_id":
                        field_type = "ObjectId"
                    else:
                        field_type = type(value).__name__
                    schema_info += f"    - {field}: {field_type}\n"

        return schema_info

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about MongoDB collections and their schemas."""
        collections = {}
        try:
            for collection_name in self.db.list_collection_names():
                # Sample documents to infer schema
                sample_docs = list(self.db[collection_name].find().limit(5))

                if sample_docs:
                    # Get all unique fields across sample documents
                    all_fields = set()
                    for doc in sample_docs:
                        all_fields.update(doc.keys())

                    # Remove ObjectId fields for display
                    sample_data = []
                    for doc in sample_docs:
                        row = []
                        for field in all_fields:
                            if field in doc:
                                if field == '_id':
                                    row.append(str(doc[field]))
                                else:
                                    row.append(doc[field])
                            else:
                                row.append(None)
                        sample_data.append(row)

                    collections[collection_name] = {
                        'columns': list(all_fields),
                        'sample_data': sample_data
                    }
                else:
                    collections[collection_name] = {
                        'columns': [],
                        'sample_data': []
                    }

            return collections
        except Exception as e:
            raise Exception(f"Error getting MongoDB collection info: {str(e)}")


class LangChainDBApp:
    def __init__(self, db_connector: DatabaseConnector, groq_api_key: str):
        """
        Initialize the LangChain DB application with the specified connector.
        """
        self.connector = db_connector
        self.connector.connect()

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

        self.query_chain = self._create_query_chain()

    def _create_query_chain(self):
        """Create the query generation chain with custom prompt."""
        if isinstance(self.connector, SQLiteConnector) or isinstance(self.connector, PostgreSQLConnector):
            # SQL-based prompt
            prompt = PromptTemplate.from_template("""
               Given the following database schema:
               {schema}

               Generate a SQL query to answer this question:
               {question}

               Rules:
               1. Use only tables and columns that exist in the schema
               2. Use syntax appropriate for {db_type}
               3. Ensure the query is optimized and efficient
               4. Return only the SQL query without any markdown formatting or explanation
               5. Do not wrap the query in code blocks
               6. If you need to modify data (UPDATE/INSERT/DELETE), always include a SELECT statement afterward to show the results
               7. Each statement must end with a semicolon

               SQL Query:
               """)

            db_type = "SQLite" if isinstance(self.connector, SQLiteConnector) else "PostgreSQL"

            chain = (
                    {
                        "schema": lambda _: self.connector.get_schema_info(),
                        "question": RunnablePassthrough(),
                        "db_type": lambda _: db_type
                    }
                    | prompt
                    | self.llm
                    | StrOutputParser()
            )




        elif isinstance(self.connector, MongoDBConnector):

            # MongoDB-based prompt

            prompt = PromptTemplate.from_template("""

               Given the following MongoDB schema:

               {schema}


               Generate a MongoDB query to answer this question:

               {question}


               Rules:

               1. Use only collections and fields that exist in the schema

               2. Return a Python dictionary that specifies the query

               3. The dictionary should have the following structure based on operation type:

                  For all operations:

                  - "collection": "collection_name"

                  - "operation": one of "find", "aggregate", "update", "insert", "delete"


                  For "find" operation:

                  - "filter": {{}} # query criteria

                  - "projection": {{}} # optional fields to return

                  - "limit": 100 # optional

                  - "include_id": False # whether to include MongoDB _id field


                  For "aggregate" operation:

                  - "pipeline": [] # aggregation pipeline stages


                  For "update" operation:

                  - "filter": {{}} # documents to update

                  - "update": {{}} # update operations

                  - "many": True/False # whether to update many documents


                  For "insert" operation:

                  - "documents": [] # list of documents to insert


                  For "delete" operation:

                  - "filter": {{}} # documents to delete

                  - "many": True/False # whether to delete many documents


               4. Return only the Python dictionary without any markdown formatting

               5. Do not wrap the dictionary in code blocks


               MongoDB Query:

               """)

            chain = (

                    {

                        "schema": lambda _: self.connector.get_schema_info(),

                        "question": RunnablePassthrough()

                    }

                    | prompt

                    | self.llm

                    | StrOutputParser()

            )

        return chain

    def _sanitize_query(self, query: str) -> str:
        """Remove markdown formatting and clean the query."""
        if not query:
            raise ValueError("Received empty query")

        # Remove markdown code block syntax if present
        query = query.replace('```sql', '').replace('```python', '').replace('```', '')
        # Remove any leading/trailing whitespace
        query = query.strip()

        if not query:
            raise ValueError("Query is empty after sanitization")

        return query

    async def generate_query(self, question: str) -> str:
        """Generate query from natural language question."""
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            generated_query = await self.query_chain.ainvoke(question)
            if not generated_query:
                raise ValueError("LLM returned empty query")

            return self._sanitize_query(generated_query)
        except Exception as e:
            raise Exception(f"Error generating query: {str(e)}")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        return self.connector.execute_query(query)

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about database tables/collections and their schemas."""
        return self.connector.get_table_info()

    def cleanup(self):
        """Clean up resources."""
        self.connector.disconnect()


async def process_query(app: LangChainDBApp, question: str):
    """Process a query asynchronously."""
    query = await app.generate_query(question)
    results = app.execute_query(query)
    return query, results


@async_handler
async def main():
    st.set_page_config(page_title="LangChain Database Explorer", layout="wide")

    load_dotenv()

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    st.sidebar.title("Configuration")

    # Database selection
    db_type = st.sidebar.radio(
        "Choose your database type",
        ["SQLite", "PostgreSQL (Neon DB)", "MongoDB Atlas"]
    )

    # Database-specific configuration
    if db_type == "SQLite":
        db_path = st.sidebar.text_input("Database Path", "test.db")
        connection_params = db_path
    elif db_type == "PostgreSQL (Neon DB)":
        db_host = st.sidebar.text_input("Host", "ep-example.region.aws.neon.tech")
        db_name = st.sidebar.text_input("Database Name", "neondb")
        db_user = st.sidebar.text_input("Username", "neon_user")
        db_password = st.sidebar.text_input("Password", type="password")
        db_port = st.sidebar.text_input("Port", "5432")
        connection_params = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    elif db_type == "MongoDB Atlas":
        mongo_uri = st.sidebar.text_input(
            "MongoDB Connection String",
            "mongodb+srv://username:password@cluster0.zcvik.mongodb.net/?retryWrites=true&w=majority"
        )
        mongo_db = st.sidebar.text_input("Database Name", "mydatabase")
        use_tls = st.sidebar.checkbox("Use TLS Certificate Verification", value=True)
        connection_params = (mongo_uri, mongo_db, use_tls)

    # API key
    groq_api_key = st.sidebar.text_input(
        "GROQ API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password"
    )

    # Connection validation
    if not connection_params or not groq_api_key:
        st.warning("Please provide database connection details and GROQ API key to continue.")
        return

    try:
        # Create appropriate connector based on selected database type
        if db_type == "SQLite":
            connector = SQLiteConnector(connection_params)
        elif db_type == "PostgreSQL (Neon DB)":
            connector = PostgreSQLConnector(connection_params)
        elif db_type == "MongoDB Atlas":
            connector = MongoDBConnector(connection_params[0], connection_params[1])

        # Initialize app with the connector
        app = LangChainDBApp(connector, groq_api_key)

    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        return

    st.title(f"LangChain {db_type} Explorer with GROQ & Llama")

    # Database explorer
    with st.expander("Database Schema"):
        table_info = app.get_table_info()

        if db_type == "MongoDB Atlas":
            for collection_name, info in table_info.items():
                st.subheader(f"Collection: {collection_name}")
                st.write("Fields:", ", ".join(info['columns']))
                st.write("Sample Data:")
                if info['sample_data']:
                    df = pd.DataFrame(info['sample_data'], columns=info['columns'])
                    st.dataframe(df)
                else:
                    st.write("No sample data available")
        else:
            for table_name, info in table_info.items():
                st.subheader(f"Table: {table_name}")
                st.write("Columns:", ", ".join(info['columns']))
                st.write("Sample Data:")
                df = pd.DataFrame(info['sample_data'], columns=info['columns'])
                st.dataframe(df)

    # Query section
    st.subheader("Ask Questions")
    question = st.text_area("Enter your question", height=100)

    if st.button("Generate and Execute Query"):
        if question:
            try:
                with st.spinner("Processing query..."):
                    query, results = await process_query(app, question)

                    st.subheader("Generated Query")
                    if db_type in ["SQLite", "PostgreSQL (Neon DB)"]:
                        st.code(query, language="sql")
                    else:
                        st.code(query, language="python")

                    st.subheader("Results")
                    st.dataframe(results)

                # Save to query history
                st.session_state.query_history.append({
                    'question': question,
                    'query': query,
                    'timestamp': pd.Timestamp.now()
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Query history
    if st.session_state.query_history:
        with st.expander("Query History"):
            for idx, item in enumerate(reversed(st.session_state.query_history)):
                st.text(f"Time: {item['timestamp']}")
                st.text(f"Question: {item['question']}")
                if db_type in ["SQLite", "PostgreSQL (Neon DB)"]:
                    st.code(item['query'], language="sql")
                else:
                    st.code(item['query'], language="python")
                st.divider()

    # Cleanup on session end
    try:
        app.cleanup()
    except:
        pass


if __name__ == "__main__":
    main()