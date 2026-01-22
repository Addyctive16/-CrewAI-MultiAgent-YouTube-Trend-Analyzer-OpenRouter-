import streamlit as st
import os
import tempfile
import gc
import time
import yaml

from tqdm import tqdm
from youtube_api_scraper import fetch_channel_videos
from dotenv import load_dotenv

# Force .env to override any system-wide environment variables
load_dotenv(override=True)

from crewai import Agent, Crew, Process, Task, LLM  # Added LLM here
from crewai_tools import FileReadTool

docs_tool = FileReadTool()

@st.cache_resource
def load_llm():
    return LLM(
        model="openrouter/openai/gpt-4o", 
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=2000,  # CRITICAL: token limit
        extra_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "YouTube Analysis App"
        }
    )

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks():
    """Creates a Crew for analysis of the channel scrapped output"""

    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize the specific OpenRouter LLM
    custom_llm = load_llm()
    
    analysis_agent = Agent(
        role=config["agents"][0]["role"],
        goal=config["agents"][0]["goal"],
        backstory=config["agents"][0]["backstory"],
        verbose=True,
        tools=[docs_tool],
        llm=custom_llm,             # Explicitly set the custom LLM
        allow_delegation=False      # Prevents spinning up default OpenAI manager
    )

    response_synthesizer_agent = Agent(
        role=config["agents"][1]["role"],
        goal=config["agents"][1]["goal"],
        backstory=config["agents"][1]["backstory"],
        verbose=True,
        llm=custom_llm,             # Explicitly set the custom LLM
        allow_delegation=False
    )

    analysis_task = Task(
        description=config["tasks"][0]["description"],
        expected_output=config["tasks"][0]["expected_output"],
        agent=analysis_agent
    )

    response_task = Task(
        description=config["tasks"][1]["description"],
        expected_output=config["tasks"][1]["expected_output"],
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[analysis_agent, response_synthesizer_agent],
        tasks=[analysis_task, response_task],
        process=Process.sequential,
        verbose=True,
        manager_llm=custom_llm,     # Force OpenRouter for manager logic
        planning_llm=custom_llm,    # Force OpenRouter for planning logic
        planning=False
    )
    return crew

# ===========================
#   Streamlit Setup
# ===========================

st.markdown("""
    # YouTube Trend Analysis powered by CrewAI & YouTube Data API
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "response" not in st.session_state:
    st.session_state.response = None

if "crew" not in st.session_state:
    st.session_state.crew = None

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def start_analysis():
    status_container = st.empty()
    
    if not os.getenv("YOUTUBE_API_KEY"):
        status_container.error("YOUTUBE_API_KEY not found.")
        return
        
    valid_channels = [ch for ch in st.session_state.youtube_channels if ch and ch.strip()]
    
    if not valid_channels:
        status_container.error("Please add at least one YouTube channel URL.")
        return
    
    with st.spinner('Scraping videos...'):
        try:
            get_transcripts = not st.session_state.get("quick_mode", False)
            channel_scrapped_output = []
            
            for ch in valid_channels:
                try:
                    vids = fetch_channel_videos(
                        ch,
                        num_videos=3,
                        start_date=st.session_state.start_date,
                        end_date=st.session_state.end_date,
                        get_transcripts=get_transcripts,
                        transcript_timeout=8.0,
                    )
                    channel_scrapped_output.extend(vids)
                except Exception as e:
                    st.warning(f"Failed to fetch from {ch}: {e}")
            
            if not channel_scrapped_output:
                st.error("No videos found.")
                return
            
            st.success(f"Found {len(channel_scrapped_output)} videos.")

            # Display Videos
            st.markdown("## YouTube Videos Extracted")
            carousel_container = st.container()
            videos_per_row = 3
            with carousel_container:
                num_videos = len(channel_scrapped_output)
                num_rows = (num_videos + videos_per_row - 1) // videos_per_row
                for row in range(num_rows):
                    cols = st.columns(videos_per_row)
                    for col_idx in range(videos_per_row):
                        video_idx = row * videos_per_row + col_idx
                        if video_idx < num_videos:
                            with cols[col_idx]:
                                st.video(channel_scrapped_output[video_idx]['url'])

            # File processing
            st.session_state.all_files = []
            os.makedirs("transcripts", exist_ok=True)
            for i in range(len(channel_scrapped_output)):
                youtube_video_id = channel_scrapped_output[i]['shortcode']
                file_path = f"transcripts/{youtube_video_id}.txt"
                with open(file_path, "w", encoding='utf-8') as f:
                    transcript_entries = channel_scrapped_output[i].get('formatted_transcript', []) or []
                    if transcript_entries:
                        for entry in transcript_entries:
                            f.write(f"({entry['start_time']:.2f}-{entry['end_time']:.2f}): {entry['text']}\n")
                    else:
                        f.write(f"Title: {channel_scrapped_output[i].get('title')}\n")
                        f.write(f"Desc: {channel_scrapped_output[i].get('description')}\n")
                st.session_state.all_files.append(file_path)

            st.session_state.channel_scrapped_output = channel_scrapped_output
            
            # Start CrewAI Analysis
            with st.spinner('Analyzing with CrewAI...'):
                st.session_state.crew = create_agents_and_tasks()
                st.session_state.response = st.session_state.crew.kickoff(
                    inputs={"file_paths": ", ".join(st.session_state.all_files)}
                )

        except Exception as e:
            st.error(f"Execution failed: {str(e)}")

# ===========================
#   Sidebar & Main UI
# ===========================
with st.sidebar:
    st.header("YouTube Channels")
    if "youtube_channels" not in st.session_state:
        st.session_state.youtube_channels = [""]
    
    def add_channel_field():
        st.session_state.youtube_channels.append("")
    
    for i, channel in enumerate(st.session_state.youtube_channels):
        col1, col2 = st.columns([6, 1])
        with col1:
            st.session_state.youtube_channels[i] = st.text_input(
                "Channel URL", value=channel, key=f"channel_{i}", label_visibility="collapsed"
            )
        with col2:
            if i > 0 and st.button("❌", key=f"remove_{i}"):
                st.session_state.youtube_channels.pop(i)
                st.rerun()
    
    st.button("Add Channel ➕", on_click=add_channel_field)
    st.divider()
    
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("Start Date").strftime("%Y-%m-%d")
    with col2:
        st.session_state.end_date = st.date_input("End Date").strftime("%Y-%m-%d")

    quick_mode = st.checkbox("⚡ Quick Mode (Skip Transcripts)", value=False)
    st.session_state.quick_mode = quick_mode
    
    st.button("Start Analysis 🚀", type="primary", on_click=start_analysis)

if st.session_state.response:
    st.markdown("### Generated Analysis")
    st.markdown(st.session_state.response)
    download_payload = str(st.session_state.response)
    st.download_button("Download Content", data=download_payload, file_name="analysis.md")

st.markdown("---")
st.markdown("Built with CrewAI & Streamlit")
