import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Optional, Any

def create_line_chart(
    data: Union[pd.DataFrame, Dict[str, List[Union[int, float]]], List[Dict[str, Union[int, float]]]],
    title: str = "Line Chart",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    color_sequence: Optional[List[str]] = None,
    height: int = 400,
    width: int = 700
) -> go.Figure:
    """
    Create a line chart using Plotly.
    
    Args:
        data: Data for the chart. Can be a pandas DataFrame, a dictionary with lists as values,
              or a list of dictionaries.
        title: Title of the chart.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        color_sequence: Optional list of colors for the lines.
        height: Height of the chart in pixels.
        width: Width of the chart in pixels.
        
    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()
    
    # Convert data to pandas DataFrame if it's not already
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a pandas DataFrame, a dictionary with lists as values, or a list of dictionaries.")
    
    # If the DataFrame has only two columns, use them as x and y
    if len(df.columns) == 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers', name=y_col))
    else:
        # Assume first column is x and the rest are y values
        x_col = df.columns[0]
        for i, col in enumerate(df.columns[1:]):
            color = color_sequence[i % len(color_sequence)] if color_sequence else None
            fig.add_trace(go.Scatter(
                x=df[x_col], 
                y=df[col], 
                mode='lines+markers', 
                name=col,
                line=dict(color=color) if color else None
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def create_bar_chart(
    data: Union[pd.DataFrame, Dict[str, List[Union[int, float]]], List[Dict[str, Union[int, float]]]],
    title: str = "Bar Chart",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    color_sequence: Optional[List[str]] = None,
    orientation: str = 'v',  # 'v' for vertical, 'h' for horizontal
    height: int = 400,
    width: int = 700
) -> go.Figure:
    """
    Create a bar chart using Plotly.
    
    Args:
        data: Data for the chart. Can be a pandas DataFrame, a dictionary with lists as values,
              or a list of dictionaries.
        title: Title of the chart.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        color_sequence: Optional list of colors for the bars.
        orientation: 'v' for vertical bars, 'h' for horizontal bars.
        height: Height of the chart in pixels.
        width: Width of the chart in pixels.
        
    Returns:
        A Plotly Figure object.
    """
    # Convert data to pandas DataFrame if it's not already
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a pandas DataFrame, a dictionary with lists as values, or a list of dictionaries.")
    
    # Create the bar chart
    if orientation == 'v':
        # If the DataFrame has only two columns, use them as x and y
        if len(df.columns) == 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=color_sequence)
        else:
            # For multiple columns, create a grouped bar chart
            fig = go.Figure()
            x_col = df.columns[0]
            for i, col in enumerate(df.columns[1:]):
                color = color_sequence[i % len(color_sequence)] if color_sequence else None
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    marker_color=color
                ))
    else:  # horizontal
        # If the DataFrame has only two columns, use them as y and x
        if len(df.columns) == 2:
            y_col = df.columns[0]
            x_col = df.columns[1]
            fig = px.bar(df, y=y_col, x=x_col, title=title, orientation='h', color_discrete_sequence=color_sequence)
        else:
            # For multiple columns, create a grouped bar chart
            fig = go.Figure()
            y_col = df.columns[0]
            for i, col in enumerate(df.columns[1:]):
                color = color_sequence[i % len(color_sequence)] if color_sequence else None
                fig.add_trace(go.Bar(
                    y=df[y_col],
                    x=df[col],
                    name=col,
                    marker_color=color,
                    orientation='h'
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        template="plotly_white",
        barmode='group'
    )
    
    return fig

def create_scatter_plot(
    data: Union[pd.DataFrame, Dict[str, List[Union[int, float]]], List[Dict[str, Union[int, float]]]],
    title: str = "Scatter Plot",
    x_label: str = "X-Axis",
    y_label: str = "Y-Axis",
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    height: int = 400,
    width: int = 700
) -> go.Figure:
    """
    Create a scatter plot using Plotly.
    
    Args:
        data: Data for the chart. Can be a pandas DataFrame, a dictionary with lists as values,
              or a list of dictionaries.
        title: Title of the chart.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        color_column: Optional column name to use for coloring points.
        size_column: Optional column name to use for sizing points.
        hover_data: Optional list of column names to include in hover information.
        height: Height of the chart in pixels.
        width: Width of the chart in pixels.
        
    Returns:
        A Plotly Figure object.
    """
    # Convert data to pandas DataFrame if it's not already
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a pandas DataFrame, a dictionary with lists as values, or a list of dictionaries.")
    
    # If the DataFrame has only two columns, use them as x and y
    if len(df.columns) == 2:
        x_col = df.columns[0]
        y_col = df.columns[1]
        fig = px.scatter(df, x=x_col, y=y_col, title=title)
    else:
        # Assume first two columns are x and y, and use additional columns for color, size, etc.
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        # Create the scatter plot
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_column if color_column and color_column in df.columns else None,
            size=size_column if size_column and size_column in df.columns else None,
            hover_data=hover_data if hover_data else None,
            title=title
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        width=width,
        template="plotly_white"
    )
    
    return fig

def detect_visualization_request(user_input: str) -> Dict[str, Any]:
    """
    Detect if the user is requesting a visualization and extract relevant information.
    
    Args:
        user_input: The user's input message.
        
    Returns:
        A dictionary containing:
        - 'is_visualization': Boolean indicating if a visualization is requested.
        - 'chart_type': The type of chart requested ('line', 'bar', 'scatter', or None).
        - 'data_description': Description of the data to visualize.
        - 'parameters': Additional parameters extracted from the request.
    """
    # Convert to lowercase for case-insensitive matching
    user_input_lower = user_input.lower()
    
    # Check for visualization keywords
    viz_keywords = ['plot', 'chart', 'graph', 'visualize', 'visualisation', 'visualization', 'display']
    is_visualization = any(keyword in user_input_lower for keyword in viz_keywords)
    
    if not is_visualization:
        return {
            'is_visualization': False,
            'chart_type': None,
            'data_description': None,
            'parameters': {}
        }
    
    # Detect chart type
    chart_type = None
    if any(term in user_input_lower for term in ['line chart', 'line graph', 'line plot']):
        chart_type = 'line'
    elif any(term in user_input_lower for term in ['bar chart', 'bar graph', 'histogram']):
        chart_type = 'bar'
    elif any(term in user_input_lower for term in ['scatter plot', 'scatter chart', 'scatter graph']):
        chart_type = 'scatter'
    
    # Extract data description
    data_description = None
    data_patterns = [
        r'(?:of|for|using|with)\s+([^.?!]+?)(?:\s+(?:by|over|across|versus|vs\.?|against))',
        r'(?:of|for|using|with)\s+([^.?!]+?)(?:\s+data)',
        r'(?:of|for|using|with)\s+([^.?!]+?)(?:\s+(?:from|in))'
    ]
    
    for pattern in data_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            data_description = match.group(1).strip()
            break
    
    # If no match found with specific patterns, try a more general approach
    if not data_description:
        # Look for text between the chart type and the end of the sentence
        chart_type_terms = ['line chart', 'bar chart', 'scatter plot', 'chart', 'graph', 'plot']
        for term in chart_type_terms:
            if term in user_input_lower:
                parts = user_input_lower.split(term, 1)
                if len(parts) > 1:
                    # Extract text after the chart type until the end of the sentence
                    after_chart_type = parts[1].strip()
                    end_sentence = re.search(r'^[^.!?]*', after_chart_type)
                    if end_sentence:
                        data_description = end_sentence.group(0).strip()
                        # Remove common prepositions at the beginning
                        data_description = re.sub(r'^(?:of|for|using|with)\s+', '', data_description)
                        break
    
    # Extract additional parameters
    parameters = {}
    
    # Title
    title_match = re.search(r'title[d:]?\s+["\']?([^"\'.?!]+)["\']?', user_input_lower)
    if title_match:
        parameters['title'] = title_match.group(1).strip()
    
    # X-axis label
    x_label_match = re.search(r'x[-\s]?(?:axis|label)[:]?\s+["\']?([^"\'.?!]+)["\']?', user_input_lower)
    if x_label_match:
        parameters['x_label'] = x_label_match.group(1).strip()
    
    # Y-axis label
    y_label_match = re.search(r'y[-\s]?(?:axis|label)[:]?\s+["\']?([^"\'.?!]+)["\']?', user_input_lower)
    if y_label_match:
        parameters['y_label'] = y_label_match.group(1).strip()
    
    return {
        'is_visualization': is_visualization,
        'chart_type': chart_type,
        'data_description': data_description,
        'parameters': parameters
    }

def generate_sample_data(data_description: str, chart_type: str) -> pd.DataFrame:
    """
    Generate sample data based on the description and chart type.
    This is a fallback when no actual data is available.
    
    Args:
        data_description: Description of the data to generate.
        chart_type: Type of chart ('line', 'bar', 'scatter').
        
    Returns:
        A pandas DataFrame with sample data.
    """
    np.random.seed(42)  # For reproducibility
    
    # Default data
    if chart_type == 'line':
        # Generate time series data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        values = np.cumsum(np.random.randn(30)) + 10
        df = pd.DataFrame({'Date': dates, 'Value': values})
        
        # Try to customize based on description
        if data_description:
            if 'temperature' in data_description or 'weather' in data_description:
                df.columns = ['Date', 'Temperature (°C)']
                df['Temperature (°C)'] = np.random.normal(20, 5, 30)
            elif 'stock' in data_description or 'price' in data_description:
                df.columns = ['Date', 'Price ($)']
                df['Price ($)'] = 100 + np.cumsum(np.random.normal(0, 2, 30))
            elif 'sales' in data_description or 'revenue' in data_description:
                df.columns = ['Date', 'Sales ($)']
                df['Sales ($)'] = 1000 + np.cumsum(np.random.normal(0, 100, 30))
            else:
                df.columns = ['Date', data_description.capitalize() if data_description else 'Value']
        
    elif chart_type == 'bar':
        # Generate categorical data
        categories = ['A', 'B', 'C', 'D', 'E']
        values = np.random.randint(10, 100, size=len(categories))
        df = pd.DataFrame({'Category': categories, 'Value': values})
        
        # Try to customize based on description
        if data_description:
            if 'sales by region' in data_description or 'regional' in data_description:
                df['Category'] = ['North', 'South', 'East', 'West', 'Central']
                df.columns = ['Region', 'Sales ($)']
            elif 'product' in data_description:
                df['Category'] = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
                df.columns = ['Product', 'Units Sold']
            elif 'age' in data_description or 'demographic' in data_description:
                df['Category'] = ['0-18', '19-35', '36-50', '51-65', '65+']
                df.columns = ['Age Group', 'Count']
            else:
                df.columns = ['Category', data_description.capitalize() if data_description else 'Value']
    
    elif chart_type == 'scatter':
        # Generate x-y data
        x = np.random.normal(0, 1, 50)
        y = x + np.random.normal(0, 0.5, 50)
        df = pd.DataFrame({'X': x, 'Y': y})
        
        # Try to customize based on description
        if data_description:
            if 'height' in data_description and 'weight' in data_description:
                df['X'] = np.random.normal(170, 10, 50)  # Heights in cm
                df['Y'] = df['X'] * 0.5 + np.random.normal(0, 5, 50)  # Weights in kg
                df.columns = ['Height (cm)', 'Weight (kg)']
            elif 'age' in data_description and ('income' in data_description or 'salary' in data_description):
                df['X'] = np.random.normal(40, 10, 50)  # Ages
                df['Y'] = df['X'] * 1000 + 20000 + np.random.normal(0, 5000, 50)  # Incomes
                df.columns = ['Age', 'Income ($)']
            elif 'study' in data_description or 'exam' in data_description:
                df['X'] = np.random.normal(5, 2, 50)  # Study hours
                df['Y'] = df['X'] * 10 + 50 + np.random.normal(0, 5, 50)  # Exam scores
                df.columns = ['Study Hours', 'Exam Score']
            else:
                x_label = 'X'
                y_label = 'Y'
                if ' vs ' in data_description:
                    parts = data_description.split(' vs ')
                    if len(parts) == 2:
                        x_label = parts[0].strip().capitalize()
                        y_label = parts[1].strip().capitalize()
                df.columns = [x_label, y_label]
    
    else:
        # Default fallback
        df = pd.DataFrame({
            'X': range(1, 11),
            'Y': np.random.randint(1, 100, 10)
        })
    
    return df