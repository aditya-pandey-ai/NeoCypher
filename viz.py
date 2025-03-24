import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import sqlite3
from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    """Specification for creating a visualization."""
    chart_type: str = Field(..., description="Type of chart (e.g., 'bar', 'line', 'pie', 'scatter')")
    title: str = Field(..., description="Chart title")
    x_column: str = Field(..., description="Column name for x-axis")
    y_column: str = Field(..., description="Column name for y-axis")
    color_column: Optional[str] = Field(None, description="Column name for color differentiation (optional)")
    aggregation: Optional[str] = Field(None, description="Aggregation function if needed (sum, avg, count)")
    sql_query: str = Field(..., description="SQL query to get the required data")


class VisualizationApp:
    def __init__(self, db_path: str, groq_api_key: str):
        self.db_path = db_path
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )
        self.viz_chain = self._create_visualization_chain()

    def _get_db_schema(self) -> str:
        """Get database schema information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        schema_info = []
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            columns_info = [f"{col[1]} ({col[2]})" for col in columns]
            schema_info.append(f"Table {table_name}:\n" + "\n".join(f"- {col}" for col in columns_info))

        conn.close()
        return "\n\n".join(schema_info)

    def _create_visualization_chain(self):
        """Create the chain for generating visualization specifications."""
        prompt = PromptTemplate.from_template("""
        Given the following database schema:
        {schema}

        Generate a visualization specification for this request: {question}

        Requirements:
        1. Choose an appropriate chart type based on the data and question
        2. Ensure the SQL query only uses existing tables and columns
        3. Use appropriate aggregations if needed
        4. Consider color coding when it adds value
        5. Provide clear and concise titles

        Return a JSON object matching this schema:
        {{
            "chart_type": "str (bar, line, pie, or scatter)",
            "title": "str",
            "x_column": "str",
            "y_column": "str",
            "color_column": "str or null",
            "aggregation": "str or null",
            "sql_query": "str"
        }}

        Response:
        """)

        class ChartSpecParser(JsonOutputParser):
            def parse(self, text):
                # Add debug print
                print(f"LLM Response: {text}")
                parsed = super().parse(text)
                return ChartSpec(**parsed)

        parser = ChartSpecParser(pydantic_object=ChartSpec)

        chain = (
                {
                    "schema": lambda _: self._get_db_schema(),
                    "question": lambda x: x
                }
                | prompt
                | self.llm
                | parser
        )

        return chain

    def get_visualization_spec(self, question: str) -> ChartSpec:
        """Generate visualization specification from natural language question."""
        try:
            result = self.viz_chain.invoke(question)
            if isinstance(result, dict):
                return ChartSpec(**result)
            return result
        except Exception as e:
            print(f"Error in visualization chain: {e}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def create_visualization(self, spec: ChartSpec, data: pd.DataFrame):
        """Create and return a Plotly figure based on the specification."""
        if spec.aggregation:
            # Convert aggregation string to the corresponding pandas function
            agg_func = {
                'avg': 'mean',
                'average': 'mean',
                'mean': 'mean',
                'sum': 'sum',
                'count': 'count',
                'min': 'min',
                'max': 'max'
            }.get(spec.aggregation.lower(), 'mean')

            # Perform the aggregation
            data = data.groupby(spec.x_column)[spec.y_column].agg(agg_func).reset_index()

        if spec.chart_type == "bar":
            fig = px.bar(
                data,
                x=spec.x_column,
                y=spec.y_column,
                color=spec.color_column,
                title=spec.title
            )
        elif spec.chart_type == "line":
            fig = px.line(
                data,
                x=spec.x_column,
                y=spec.y_column,
                color=spec.color_column,
                title=spec.title
            )
        elif spec.chart_type == "pie":
            fig = px.pie(
                data,
                names=spec.x_column,
                values=spec.y_column,
                title=spec.title
            )
        elif spec.chart_type == "scatter":
            fig = px.scatter(
                data,
                x=spec.x_column,
                y=spec.y_column,
                color=spec.color_column,
                title=spec.title
            )
        else:
            raise ValueError(f"Unsupported chart type: {spec.chart_type}")

        return fig


def main():
    st.set_page_config(page_title="Natural Language Data Visualization", layout="wide")

    st.title("Natural Language Data Visualization")

    # Sidebar configuration
    st.sidebar.title("Configuration")
    db_path = st.sidebar.text_input("Database Path", "path/to/your/database.db")
    groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")

    if not db_path or not groq_api_key:
        st.warning("Please provide database path and GROQ API key to continue.")
        return

    try:
        app = VisualizationApp(db_path, groq_api_key)
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        return

    # Main interface
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

                    # Show the generated SQL
                    st.subheader("Generated SQL Query")
                    st.code(spec.sql_query, language="sql")

                    # Execute query and get data
                    data = app.execute_query(spec.sql_query)

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


if __name__ == "__main__":
    main()