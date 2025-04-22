import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field, validator
import os
# Database Connectors
import sqlite3
import pymongo
import psycopg2
from sqlalchemy import create_engine


class DatabaseConfig(BaseModel):
    """Configuration for different database types"""
    db_type: str = Field(..., description="Type of database (sqlite, mongodb, postgresql)")
    connection_string: str = Field(..., description="Connection string or path")
    database_name: Optional[str] = Field(None, description="Database name (for MongoDB/PostgreSQL)")

    @validator('db_type')
    def validate_db_type(cls, v):
        valid_types = ['sqlite', 'mongodb', 'postgresql']
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid database type. Must be one of {valid_types}")
        return v.lower()

    @validator('database_name')
    def validate_database_name(cls, v, values):
        # Only require database_name for MongoDB
        if values.get('db_type') == 'mongodb' and not v:
            raise ValueError("Database name is required for MongoDB")
        return v

class DatabaseConnector:
    """Unified database connection and querying interface"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self._connect()

    def _connect(self):
        """Establish database connection based on type"""
        try:
            if self.config.db_type == 'sqlite':
                self.connection = sqlite3.connect(self.config.connection_string)

            elif self.config.db_type == 'mongodb':
                if not self.config.database_name:
                    raise ValueError("Database name is required for MongoDB")
                self.connection = pymongo.MongoClient(self.config.connection_string)
                self.database = self.connection[self.config.database_name]

            elif self.config.db_type == 'postgresql':
                self.connection = create_engine(self.config.connection_string)
        except Exception as e:
            raise ConnectionError(f"Database connection error: {str(e)}")

    def get_schema(self) -> str:
        """Retrieve database schema information"""
        try:
            if self.config.db_type == 'sqlite':
                return self._get_sqlite_schema()

            elif self.config.db_type == 'mongodb':
                return self._get_mongodb_schema()

            elif self.config.db_type == 'postgresql':
                return self._get_postgresql_schema()
        except Exception as e:
            raise ValueError(f"Error retrieving schema: {str(e)}")

    def _get_sqlite_schema(self) -> str:
        cursor = self.connection.cursor()
        schema_info = []
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            columns_info = [f"{col[1]} ({col[2]})" for col in columns]
            schema_info.append(f"Table {table_name}:\n" + "\n".join(f"- {col}" for col in columns_info))

        return "\n\n".join(schema_info)

    def _get_mongodb_schema(self) -> str:
        schema_info = []
        for collection_name in self.database.list_collection_names():
            collection = self.database[collection_name]
            # Get a sample document to infer schema
            sample = collection.find_one()
            if sample:
                fields = [f"{k}: {type(v).__name__}" for k, v in sample.items()]
                schema_info.append(f"Collection {collection_name}:\n" + "\n".join(f"- {field}" for field in fields))
        return "\n\n".join(schema_info)

    def _get_postgresql_schema(self) -> str:
        query = """
                    SELECT 
                        table_name, 
                        column_name, 
                        data_type,
                        is_nullable,
                        column_default
                    FROM 
                        information_schema.columns 
                    WHERE 
                        table_schema = 'public'
                    ORDER BY 
                        table_name, ordinal_position
                    """

        try:
            # Use pandas to fetch schema details
            schema_df = pd.read_sql(query, self.connection)

            # Organize schema information
            schema_info = {}
            for table in schema_df['table_name'].unique():
                table_columns = schema_df[schema_df['table_name'] == table]

                # Create a dictionary of column details
                columns = {}
                for _, row in table_columns.iterrows():
                    columns[row['column_name']] = {
                        'type': row['data_type'],
                        'nullable': row['is_nullable'] == 'YES',
                        'default': row['column_default']
                    }

                schema_info[table] = columns

            return schema_info

        except Exception as e:
            print(f"Error extracting schema: {e}")
            return {}

    def execute_query(self, query: str) -> pd.DataFrame:
        """Enhanced query execution with type detection"""
        try:
            if self.config.db_type == 'sqlite':
                return pd.read_sql_query(query, self.connection)

            elif self.config.db_type == 'mongodb':

                # Parse the query as a generic aggregation pipeline

                try:

                    # Try to parse the query as a custom aggregation pipeline

                    try:

                        # Attempt to parse as JSON aggregation pipeline

                        pipeline = json.loads(query)

                    except json.JSONDecodeError:

                        # If not JSON, try to dynamically generate pipeline

                        # Extract collection and grouping logic from query

                        import re

                        # Extract collection name (assuming it's mentioned in the query)

                        collection_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)

                        if not collection_match:
                            raise ValueError("Could not determine collection name")

                        collection_name = collection_match.group(1)

                        # Extract grouping column

                        group_match = re.search(r'GROUP\s+BY\s+(\w+)', query, re.IGNORECASE)

                        group_column = group_match.group(1) if group_match else None

                        # Default aggregation pipeline

                        pipeline = [

                            {"$group": {

                                "_id": f"${group_column}" if group_column else None,

                                "count": {"$sum": 1}

                            }},

                            {"$project": {

                                "group_key": "$_id",

                                "count": 1,

                                "_id": 0

                            }}

                        ]

                    # Execute aggregation

                    collection = self.database[collection_name]

                    cursor = collection.aggregate(pipeline)

                    # Convert to DataFrame

                    result = pd.DataFrame(list(cursor))

                    # Rename columns if needed

                    if 'group_key' in result.columns:
                        result = result.rename(columns={'group_key': group_column}) if group_column else result

                    return result


                except Exception as mongo_error:

                    print(f"MongoDB query error: {mongo_error}")

                    raise ValueError(f"Invalid MongoDB query: {mongo_error}")

            elif self.config.db_type == 'postgresql':
                return pd.read_sql(query, self.connection)

        except Exception as e:
            raise ValueError(f"Query execution error: {str(e)}")

class ChartSpec(BaseModel):
    """Specification for creating a visualization."""
    chart_type: str = Field(..., description="Type of chart (e.g., 'bar', 'line', 'pie', 'scatter')")
    title: str = Field(..., description="Chart title")
    x_column: str = Field(..., description="Column name for x-axis")
    y_column: str = Field(..., description="Column name for y-axis")
    color_column: Optional[str] = Field(None, description="Column name for color differentiation (optional)")
    aggregation: Optional[str] = Field(None, description="Aggregation function if needed (sum, avg, count)")
    sql_query: str = Field(..., description="Query to get the required data")

class VisualizationApp:
    def __init__(self, db_connector, groq_api_key: str):
        self.db_connector = db_connector
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )
        self.viz_chain = self._create_visualization_chain()

    def _create_visualization_chain(self):
        """Create the chain for generating visualization specifications."""
        # Get database schema
        schema = self.db_connector.get_schema()

        # Create prompt template with dynamic schema
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data visualization assistant. 
                    Given the following database schema:
                    {schema}

                    Your task is to generate a precise visualization specification based on the user's request.

                    Visualization Specification Requirements:
                    1. Carefully review the actual table and column names in the schema
                    2. Use EXACT column names from the schema
                    3. Use EXACT column names from the schema
                    4. Select meaningful columns for x and y axes
                    - For pie charts, use a categorical column as x_column
                    - Ensure x_column is not null
                    5. Use aggregation if necessary
                    6. Create a descriptive title
                    7. Construct an accurate query to retrieve the data
                    8. Carefully review the actual table and column names in the schema
                    9. Use EXACT column names from the schema
                    10. Choose the most appropriate chart type
                    11. Select meaningful columns for aggregation
                    12. Create a descriptive title
                    13. Construct an accurate query using EXACT column names

                    Response Format (MUST be a valid JSON):
                    {{
                        "chart_type": "bar|line|pie|scatter",
                        "title": "Descriptive chart title",
                        "x_column": "column name for x-axis",
                        "y_column": "column name for y-axis",
                        "color_column": "optional column for color coding",
                        "aggregation": "optional aggregation method (sum/avg/count/min/max)",
                        "sql_query": "SQL query to retrieve the data"
                    }}
                    """),
            ("human", "Database Schema: {schema}\n\nVisualization Request: {question}")
        ])

        parser = parser = PydanticOutputParser(pydantic_object=ChartSpec)

        chain = (
                prompt
                | self.llm
                | parser
        )

        return chain

    def get_visualization_spec(self, question: str) -> ChartSpec:
        """Generate visualization specification from natural language question."""
        try:
            # Get the database schema
            schema = self.db_connector.get_schema()

            # Invoke the chain with schema and question
            result = self.viz_chain.invoke({
                "schema": schema,
                "question": question
            })

            return result
        except Exception as e:
            print(f"Error in visualization chain: {e}")
            raise

    def create_visualization(self, spec: ChartSpec, data: pd.DataFrame):
        """Create and return a Plotly figure based on the specification."""
        # Debug print initial state
        print("Initial Data Columns:", data.columns)
        print("Spec Details:", spec.dict())

        # Flexible column name matching
        def find_column(target_column):
            # Case-insensitive exact match
            exact_matches = [col for col in data.columns if col.upper() == target_column.upper()]
            if exact_matches:
                return exact_matches[0]

            # Partial match
            partial_matches = [col for col in data.columns if target_column.upper() in col.upper()]
            if partial_matches:
                return partial_matches[0]

            # If no match found
            raise ValueError(f"Cannot find column matching: {target_column}. Available columns: {list(data.columns)}")

        try:
            # Validate and find correct column names
            x_column = find_column(spec.x_column)
            y_column = find_column(spec.y_column)

            data[y_column] = pd.to_numeric(data[y_column], errors='coerce')

            # Aggregation logic with error handling
            if spec.aggregation:
                agg_func = {
                    'avg': 'mean',
                    'average': 'mean',
                    'mean': 'mean',
                    'sum': 'sum',
                    'count': 'count',
                    'min': 'min',
                    'max': 'max'
                }.get(spec.aggregation.lower(), 'mean')

                try:
                    # Perform aggregation
                    data = data.groupby(x_column)[y_column].agg(agg_func).reset_index()
                except Exception as agg_error:
                    print(f"Aggregation error: {agg_error}")
                    # Fallback to original data if aggregation fails

            # Ensure columns are of appropriate type
            try:
                data[x_column] = data[x_column].astype(str)
                data[y_column] = pd.to_numeric(data[y_column], errors='coerce')
            except Exception as type_error:
                print(f"Type conversion error: {type_error}")

            # Color column handling
            color_column = None
            if spec.color_column:
                try:
                    color_column = find_column(spec.color_column)
                except ValueError:
                    print(f"Could not find color column: {spec.color_column}")
                    color_column = None

            # Visualization creation logic with enhanced error handling
            if spec.chart_type == "bar":
                fig = px.bar(
                    data,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    title=spec.title
                )
            elif spec.chart_type == "line":
                fig = px.line(
                    data,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    title=spec.title
                )
            elif spec.chart_type == "pie":
                fig = px.pie(
                    data,
                    names=x_column,
                    values=y_column,
                    title=spec.title
                )
                fig.update_traces(texttemplate='%{label}<br>%{value} (%{percent})')
            elif spec.chart_type == "scatter":
                fig = px.scatter(
                    data,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    title=spec.title
                )
            else:
                raise ValueError(f"Unsupported chart type: {spec.chart_type}")

            return fig

        except Exception as e:
            print(f"Visualization creation error: {e}")
            # Provide more context about the error
            print(f"Data columns: {list(data.columns)}")
            print(f"Spec details: {spec.dict()}")
            raise


def main():
    st.set_page_config(page_title="Multi-Database Visualization", layout="wide")
    st.title("Multi-Database Natural Language Visualization")

    # Sidebar configuration
    st.sidebar.title("Database Configuration")

    # Database Type Selection
    db_type = st.sidebar.selectbox(
        "Select Database Type",
        ['SQLite', 'MongoDB', 'PostgreSQL']
    )

    # Dynamic connection input based on database type
    if db_type == 'SQLite':
        connection_string = st.sidebar.text_input("SQLite Database Path", "path/to/database.db")
        database_name = None
    elif db_type == 'MongoDB':
        connection_string = st.sidebar.text_input(
            "MongoDB Connection String",
            "mongodb+srv://username:password@cluster.mongodb.net/"
        )
        database_name = st.sidebar.text_input("Database Name", key="mongodb_db_name")
    else:  # PostgreSQL
        connection_string = st.sidebar.text_input(
            "PostgreSQL Connection String",
            "postgresql://username:password@host:port/database"
        )
        database_name = None

    # GROQ API Key
    groq_api_key = st.sidebar.text_input(
        "GROQ API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password"
    )
    # Validate inputs
    if not connection_string or not groq_api_key:
        st.warning("Please provide connection details and GROQ API key")
        return

    # Create database configuration
    try:
        db_config = DatabaseConfig(
            db_type=db_type.lower(),
            connection_string=connection_string,
            database_name=database_name
        )

        # Initialize database connector
        db_connector = DatabaseConnector(db_config)

        # Initialize app
        app = VisualizationApp(db_connector, groq_api_key)

        # Main visualization interface
        question = st.text_area(
            "What would you like to visualize?",
            placeholder="e.g., 'Show me a bar chart of total sales by month'"
        )

        if st.button("Generate Visualization"):
            if question:
                try:
                    with st.spinner("Generating visualization..."):
                        # Get visualization specification
                        spec = app.get_visualization_spec(question)

                        # Show the generated Query
                        st.subheader("Generated Query")
                        st.code(spec.sql_query, language="sql")

                        # Execute query and get data
                        data = db_connector.execute_query(spec.sql_query)

                        # Show the data
                        st.subheader("Data Preview")
                        st.dataframe(data.head())

                        # Create and display visualization
                        st.subheader("Visualization")
                        fig = app.create_visualization(spec, data)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a visualization request.")

    except Exception as e:
        st.error(f"Configuration Error: {str(e)}")


if __name__ == "__main__":
    main()