# app.py
from typing import List, Dict, Any
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


def async_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class LangChainSQLApp:
    def __init__(self, db_path: str, groq_api_key: str):
        """
        Initialize the LangChain SQL application.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

        self.db_path = db_path
        try:
            self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            # Verify connection works
            with sqlite3.connect(db_path) as conn:
                conn.cursor().execute("SELECT 1")
        except Exception as e:
            raise Exception(f"Failed to initialize database connection: {str(e)}")

        self.sql_chain = self._create_sql_chain()

    def _sanitize_sql_query(self, query: str) -> str:
        """Remove markdown formatting and clean the SQL query."""
        if not query:
            raise ValueError("Received empty query")

        # Remove markdown code block syntax if present
        query = query.replace('```sql', '').replace('```', '')
        # Remove any leading/trailing whitespace
        query = query.strip()

        if not query:
            raise ValueError("Query is empty after sanitization")

        return query

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

    def _create_sql_chain(self):
        """Create the SQL generation chain with custom prompt."""
        prompt = PromptTemplate.from_template("""
           Given the following database schema:
           {schema}

           Generate a SQL query to answer this question:
           {question}

           Rules:
           1. Use only tables and columns that exist in the schema
           2. Use SQLite syntax
           3. Ensure the query is optimized and efficient
           4. Return only the SQL query without any markdown formatting or explanation
           5. Do not wrap the query in code blocks
           6. If you need to modify data (UPDATE/INSERT/DELETE), always include a SELECT statement afterward to show the results
           7. Each statement must end with a semicolon

           SQL Query:
           """)

        chain = (
                {
                    "schema": lambda _: self.db.get_table_info(),
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return chain

    async def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question."""
        if not question:
            raise ValueError("Question cannot be empty")

        try:
            sql_query = await self.sql_chain.ainvoke(question)
            if not sql_query:
                raise ValueError("LLM returned empty query")

            return self._sanitize_sql_query(sql_query)
        except Exception as e:
            raise Exception(f"Error generating SQL: {str(e)}")

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        if not sql_query:
            raise ValueError("SQL query cannot be empty")

        try:
            # Sanitize the query before execution
            clean_query = self._sanitize_sql_query(sql_query)

            # Split into individual statements
            statements = self._split_sql_statements(clean_query)
            if not statements:
                raise ValueError("No valid SQL statements found")

            conn = sqlite3.connect(self.db_path)
            try:
                # Start a transaction
                conn.execute("BEGIN TRANSACTION")

                # Execute all non-SELECT statements first
                cursor = conn.cursor()
                for stmt in statements[:-1]:  # All but the last statement
                    if stmt.strip().upper().startswith('SELECT'):
                        continue  # Skip SELECT statements except the last one
                    cursor.execute(stmt)

                # Commit the transaction
                conn.commit()

                # Execute the final statement (should be a SELECT) and return results
                final_stmt = statements[-1]
                if not final_stmt.strip().upper().startswith('SELECT'):
                    # If last statement isn't a SELECT, execute it and return empty DataFrame
                    cursor.execute(final_stmt)
                    conn.commit()
                    return pd.DataFrame()

                # Return results of the final SELECT statement
                df = pd.read_sql_query(final_stmt, conn)
                return df if not df.empty else pd.DataFrame()

            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()

        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about database tables and their schemas."""
        tables = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

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
        finally:
            if 'conn' in locals():
                conn.close()

async def process_query(app: LangChainSQLApp, question: str):
    """Process a query asynchronously."""
    sql_query = await app.generate_sql(question)
    results = app.execute_query(sql_query)
    return sql_query, results


@async_handler
async def main():
    st.set_page_config(page_title="LangChain SQL Explorer", layout="wide")

    load_dotenv()

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    st.sidebar.title("Configuration")
    db_path = st.sidebar.text_input("Database Path", "path/to/your/database.db")
    groq_api_key = st.sidebar.text_input(
        "GROQ API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password"
    )

    if not db_path or not groq_api_key:
        st.warning("Please provide database path and GROQ API key to continue.")
        return

    try:
        app = LangChainSQLApp(db_path, groq_api_key)
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        return

    st.title("LangChain SQL Explorer with GROQ & Llama")

    # Database explorer
    with st.expander("Database Schema"):
        table_info = app.get_table_info()
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
                    sql_query, results = await process_query(app, question)

                    st.subheader("Generated SQL")
                    st.code(sql_query, language="sql")

                    st.subheader("Results")
                    st.dataframe(results)

                # Save to query history
                st.session_state.query_history.append({
                    'question': question,
                    'sql_query': sql_query,
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
                st.code(item['sql_query'], language="sql")
                st.divider()


if __name__ == "__main__":
    main()