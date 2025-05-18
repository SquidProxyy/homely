import streamlit as st

st.title("Real Estate AVM Dashboard - Test")
st.write("Testing library imports...")

# Test basic imports first
st.write("Importing pandas and numpy...")
import pandas as pd
import numpy as np
st.success("✅ pandas and numpy successfully imported!")

# Test matplotlib separately 
st.write("Importing matplotlib...")
try:
    import matplotlib.pyplot as plt
    st.success("✅ matplotlib successfully imported!")
    
    # Create a simple matplotlib plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
    st.pyplot(fig)
except Exception as e:
    st.error(f"❌ matplotlib error: {str(e)}")

# Test all other imports
st.write("Importing other libraries...")
try:
    import seaborn as sns
    import plotly.express as px
    # Only import what's needed
    import joblib
    st.success("✅ All other libraries successfully imported!")
except Exception as e:
    st.error(f"❌ Error importing other libraries: {str(e)}")

st.write("System info:")
import sys
st.info(f"Python version: {sys.version}")
