"""Streamlit frontend for viewing agent metrics."""
import streamlit as st
import requests

# Configure the page
st.set_page_config(
    page_title="Multi-Agent Financial Analysis",
    layout="wide"
)

# Title and description
st.title("Multi-Agent Financial Analysis System")
st.write("Enter a stock ticker to get analysis from three different perspectives.")

# Input for ticker symbol
ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()

if st.button("Analyze"):
    try:
        # Call FastAPI backend
        response = requests.get(f"http://localhost:8000/analyze/{ticker}")
        if response.status_code == 200:
            data = response.json()

            # Display the Judge's final verdict first
            if "judge" in data and data["judge"]:
                st.subheader("🏆 Final Verdict from the Judge")
                if "error" in data["judge"]:
                    st.error(f"Judge Analysis Error: {data['judge']['error']}")
                else:
                    st.markdown(data["judge"]["analysis"])
                st.write("---_"*10)  # A more prominent separator
            
            # Display analysis from each agent
            st.subheader("Individual Agent Analyses")
            for agent_name, agent_data in data["agents"].items():
                st.markdown(f"**🤖 {agent_name}**")
                
                # Show the agent's analysis
                if "error" in agent_data:
                    st.error(f"Analysis Error: {agent_data['error']}")
                else:
                    st.markdown(agent_data["analysis"])

                # Show raw metrics in an expander
                with st.expander("View Raw Metrics"):
                    for metric, value in agent_data["raw_metrics"].items():
                        if isinstance(value, float):
                            st.write(f"{metric}: {value:.2f}")
                        else:
                            st.write(f"{metric}: {value}")
                
                st.write("---")  # Add a separator between agents
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")