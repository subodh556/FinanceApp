__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit  as st
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
import time
from dotenv import load_dotenv , find_dotenv
from crewai import LLM





# os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
_ = load_dotenv(find_dotenv())

#run this command in the terminal to login to the huggingface platform -> huggingface-cli login --token 'your access token' --add-to-git-credential

os.environ["OPENAI_API_KEY"] = "dummy-key"

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"] 

chatm = LLM(
    model="huggingface/mistralai/Mistral-7B-Instruct-v0.3",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_tokens=500,
)

#Streamlit app configuration

st.set_page_config(page_title="AI Finance App", page_icon="💸", layout="wide")

#UI Title
st.title("AI Agent for Financial Analysis")
st.write("Analyze stock trends and make data-driven investment decisions with AI.")

#Sidebar User Inputs
st.sidebar.header("Enter Company Details")
company_name = st.sidebar.text_input("Company Name", "Tata , Inc.")

#Button to start Analysis
start_analysis = st.sidebar.button("🔍 Start Analysis")

#Thinking Section (Progress Tracker)
thinking_section = st.empty()

if start_analysis:
    with thinking_section.container():
        st.subheader("🧠 Thinking... AI is working on financial analysis")
        progress_bar = st.progress(0)

        #define financial analysis agent
        st.write("➡️ Setting up **Financial Analyst** agent...")
        time.sleep(1)
        financial_analyst = Agent(
            role = "Senior Financial Analyst",
            goal = "Analyze market trends and provide accurate stock investment insights.",
            backstory= (
                "You are a seasoned financial analyst with expertise in stock market trends, "
                "earnings reports, and financial modeling. Your job is to help investors make "
                "informed decisions by providing deep market insights."
            ),
            allow_delegation= False,
            llm=chatm,
            verbose= True,
        )
        progress_bar.progress(20)

        # Define Investment Quality Assurance Agent
        st.write("➡️ Setting up **Investment Strategy Reviewer** agent...")
        time.sleep(1)
        investment_qc = Agent(
            role="Investment Strategy Reviewer",
            goal="Ensure financial reports meet the highest accuracy and reliability standards.",
            backstory=(
                "You are an experienced investment strategist responsible for reviewing "
                "financial reports and ensuring investment recommendations are well-founded, "
                "clear, and based on reliable data sources."
            ),
            llm=chatm,
            verbose=True
        )
        progress_bar.progress(40)

        #Financial Data Scraping Tool
        st.write("➡️ Setting up **Stock Market Scraper Tool**...")
        time.sleep(1)
        finance_scraper_tool = ScrapeWebsiteTool(
            website_url="https://www.nasdaq.com/market-activity/stocks"
        )
        progress_bar.progress(60)

        # Define Financial Analysis Task
        st.write("➡️ Creating **Stock Market Analysis Task**...")
        time.sleep(1)
        stock_analysis_task = Task(
            description=(
                f"Analyze the latest earnings report and stock trends for {company_name}. "
                "Provide insights on whether the stock is a good investment, considering "
                "market conditions, financial performance, and risk factors."
            ),
            expected_output=(
                f"A detailed financial analysis report on {company_name}, including key "
                "financial metrics, stock performance trends, and a well-reasoned "
                "investment recommendation (Buy/Hold/Sell)."
            ),
            tools=[finance_scraper_tool],
            agent=financial_analyst,
            max_retries=1,
        )
        progress_bar.progress(80)

        # Define Investment Review Task
        st.write("➡️ Creating **Investment Review Task**...")
        time.sleep(1)
        investment_review_task = Task(
            description=(
                f"Review the financial analysis report for {company_name}. Ensure that "
                "all data is accurate, sources are properly cited, and the investment "
                "recommendation is justified."
            ),
            expected_output=(
                f"A refined and well-reviewed financial report for {company_name}, ensuring "
                "clarity, accuracy, and completeness."
            ),
            agent=investment_qc,
            max_retries=1,

        )
        progress_bar.progress(90)

        # Define Crew
        st.write("➡️ Creating **Financial Analysis Crew** and running tasks...")
        time.sleep(1)
        finance_crew = Crew(
            agents = [financial_analyst, investment_qc],
            tasks = [stock_analysis_task, investment_review_task],
            verbose = True,
            memory=True
        )

        # Run CrewAI Process
        st.write("🚀 **AI is now analyzing stock market trends...**")
        result = finance_crew.kickoff(inputs = {"company_name": company_name})
        progress_bar.progress(100)

        # Display Final Markdown Report
        st.subheader("📊 **Final Financial Analysis Report**")
        st.markdown(result)