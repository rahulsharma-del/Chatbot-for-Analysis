# app.py (minimal test version)

import streamlit as st
import pandas as pd

# Import local analytics module
import analytics as A  # make sure analytics.py is in the same folder

def main():
    st.title("🚀 Minimal Test App")
    st.write("If you see this, the indentation error is fixed 🎉")

    # Quick test: call a simple function from analytics.py if available
    if hasattr(A, "daily_metrics"):
        st.success("Analytics module imported successfully!")
    else:
        st.warning("Analytics module loaded, but daily_metrics() not found.")

if __name__ == "__main__":
    main()



