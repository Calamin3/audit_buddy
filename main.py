import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from crewai_tools import PDFSearchTool, SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM

# Load environment variables
load_dotenv('.env')
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

#Initialize LLM using Govtext LLM

llm = LLM(
    model="openai/"+os.getenv("OPENAI_MODEL_NAME"), 
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://litellm.govtext.gov.sg/",
    default_headers={"user-agent":"Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/81.0"}
)

# Initialize tools
tool_pdf = PDFSearchTool(
    config=dict(
        embedder=dict(
            provider="openai",
            config=dict(
                model="text-embedding-3-large-prd-gcc2-lb"
            ),
        ),
    )
)

tool_search = SerperDevTool(n_results=8, country="sg")
tool_webscrape = ScrapeWebsiteTool()

def extract_file_paths(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        with open(os.path.join("tempdir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(os.path.join("tempdir", uploaded_file.name))
    return file_paths

# Streamlit app
st.title("Audit Buddy")

# User input for topic
topic = st.text_input("Enter the audit topic:")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

#Select useful website to search for
domain = st.multiselect(
    "Focused websites to search through",
    ["Hansard SG","Gov.sg","Reddit","Hardwarezone Forums"],
    ["Hansard SG", "Gov.sg"],
)

if st.button("Let's Go!!!") and topic and uploaded_files:
    # Create a temporary directory to store uploaded files
    os.makedirs("tempdir", exist_ok=True)
    
    # Extract file paths
    file_paths = extract_file_paths(uploaded_files)

    # Define agents and tasks (same as before)
    auditor = Agent(
        role="auditor",
        goal=f"Generate the Audit Planning Memo on {topic} based on inputs from researcher and past audit findings and financial information from the audit assistant.",
        backstory=f"You are an auditor. You need to consider the inputs from the researcher and audit assistant to determine what to write for the audit planning memo and risk assessment",
        allow_delegation=True,
        max_iter=15,
        verbose=True,
        llm = llm
    )

    researcher = Agent(
        role="researcher",
        goal=f"""From the internet, research and analyse and sieve out information related to {topic} only in the Singapore context. 
        Think step by step and extract : regulatory information, potential audit findings, fraud cases and reasons why.""",
        backstory=f"As a researcher, navigating and extracting critical information is crucial. You are assisting the auditor to research on top 10 sources most relevant to {topic} in the Singapore context.",
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        llm = llm
    )

    audit_assistant = Agent(
        role="audit assistant",
        goal=f"From PDF, extract the audit findings or financial information pertaining to {topic} only.",
        backstory=f"You are an audit assistant. Your job is to extract the past audit findings or financial statements relating to the {topic}. Output could either be in text for past audit findings or JSON for financial statements.",
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        llm = llm
    )

    task_search = Task(
        description=f"Search for the top websites that contain the most relevant information, in the context of Singapore, that are related to regulations on {topic} only.",
        expected_output=f"Returns a list of top websites that contain most relevant regulations related information, in the context of Singapore, on {topic} only. The output must be in JSON format with the description as the key and url as value.",
        agent=researcher,
        tools=[tool_search],
    )

    task_focused_search = Task(
        description=f"Search for websites from {domain} that contain the most relevant information, in the context of Singapore, that are related to regulations on {topic} only.",
        expected_output=f"Returns a list of top websites from {domain} that contain most relevant information, in the context of Singapore, on {topic} only. The output must be in JSON format with the description as the key and url as value.",
        agent=researcher,
        tools=[tool_search],
    )

    task_websitesearch = Task(
        description=f"Scrape websites for all regulations and information related to {topic} only. For urls ending with .PDF, use readpdf tool",
        expected_output=f"A markdown document with a summary of information related to {topic}. Include the sources (description and URL/filename) used.",
        agent=researcher,
        tools=[tool_webscrape,tool_pdf],
        context=[task_search,task_focused_search]
    )

    task_readpdf = Task(
        description=f"For each uploaded file path in {file_paths}, read the pdf and extract all audit findings pertaining to {topic}. Provide citation",
        expected_output=f"A list of audit findings pertaining to {topic}",
        tools=[tool_pdf],
        agent=audit_assistant
    )

    task_write = Task(
        description=f"""
        1. Use the content from web searches and files to write an audit planning memo and risk assessment based on {topic} for the team to work on.
        2. Provide the background of the audit topic using the context provided
        3. Include the sources used - websites URLs and file names of files used.
        4. Proofread for grammatical errors and alignment with the common style used in audit planning memos.""",
        expected_output="""
        A well-written audit planning memo and risk assessment in markdown format.""",
        agent=auditor,
        context=[task_websitesearch, task_readpdf],
        output_file="APM.md"
    )

    crew = Crew(
        agents=[auditor, researcher, audit_assistant],
        tasks=[task_search, task_focused_search, task_websitesearch, task_readpdf, task_write],
        verbose=True
    )

    with st.spinner("Generating Audit Planning Memo..."):
        result = crew.kickoff(inputs={"topic": topic, "file_paths": file_paths,"domain": domain})

    st.success("Audit Planning Memo generated successfully!")

    # Display results
    st.subheader("Raw Output")
    st.text(result.raw)

    st.subheader("Token Usage")
    st.text(result.token_usage)

    for i in range(len(result.tasks_output[:-1])):
        st.subheader(f"Task {i+1} Output")
        st.text(result.tasks_output[i])

    st.subheader(f"Final Output")
    st.text(result.tasks_output[-1])

    if os.path.exists("APM.md"):
        with open("APM.md", "r") as f:
            apm_content = f.read()
            st.download_button(
            label="Download Audit Planning Memo",
            data=apm_content,
            file_name="Audit_Planning_Memo.md",
            mime="text/markdown"
            )

    else:
        st.warning("APM.md file not found. The memo might not have been generated successfully.")

    # Clean up temporary files
    for file_path in file_paths:
        os.remove(file_path)
    os.rmdir("tempdir")

else:
    st.warning("Please enter an audit topic and upload at least one PDF file.")