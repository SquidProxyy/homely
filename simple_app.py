import streamlit as st
import pandas as pd
import numpy as np

st.title("Real Estate AVM Dashboard - Simple Version")
st.write("Testing basic functionality...")

# Test matplotlib import
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a simple matplotlib plot
    st.subheader("Matplotlib Test")
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x))
    st.pyplot(fig)
    
    st.success("✅ Matplotlib is working!")
except Exception as e:
    st.error(f"❌ Matplotlib error: {str(e)}")

# Test plotly
try:
    import plotly.express as px
    
    st.subheader("Plotly Test")
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    st.plotly_chart(fig)
    
    st.success("✅ Plotly is working!")
except Exception as e:
    st.error(f"❌ Plotly error: {str(e)}")

# Test if basic dataframe manipulation works
st.subheader("DataFrame Test")
data = {
    'property_id': [1001, 1002, 1003],
    'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd'],
    'price': [750000, 950000, 620000]
}
df = pd.DataFrame(data)
st.dataframe(df)

st.subheader("System Info")
import sys
st.write(f"Python version: {sys.version}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"NumPy version: {np.__version__}")
try:
    st.write(f"Matplotlib version: {plt.__version__}")
except:
    st.write("Matplotlib not available")

st.success("Basic test complete!")
