import streamlit as st
import re
from transformers import pipeline
from huggingface_hub import login
from youtube_transcript_api import YouTubeTranscriptApi
from serpapi import GoogleSearch

HF_TOKEN = st.secrets["HF_API_KEY"]
Serp_api_key = st.secrets["SERP_API_KEY"]

login(token=HF_TOKEN, new_session=False)

@st.cache_resource
def load_pipeline():
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        token=HF_TOKEN,
        max_length=2048,
        do_sample=True,
        temperature=0.7
    )

pipe = load_pipeline()

def search_evidence(claim, api_key):
    params = {
        "engine": "google",
        "q": f"{claim} site:mayoclinic.org OR site:cdc.gov OR site:nih.gov OR site:medlineplus.gov",
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    sources = []
    for result in results.get("organic_results", [])[:3]:
        title = result.get("title", "No Title")
        link = result.get("link", "No Link")
        snippet = result.get("snippet", "No Description available.")
        formatted_result = f"Title: {title}\nLink: {link}\nDescription: {snippet}\n"
        sources.append(formatted_result)
    return sources

st.title("YouTube Medical Claim Fact Checker")
yt_url = st.text_input("Paste a YouTube video URL:")

if yt_url:
    vid_id = yt_url.split("v=")[-1]
    fetch_ytt_vid = YouTubeTranscriptApi.get_transcript(vid_id, languages=['en'])

    fulltext = " ".join(entry['text'] for entry in fetch_ytt_vid)
    revised_text = re.sub(r'\[.*?\]', '', fulltext)
    revised_text = re.sub(r'\s+', ' ', revised_text)
    revised_text = revised_text.lower()

    st.subheader("Transcript Sample")
    st.write(revised_text[:1000] + "...")

    prompt = f"""<s>[INST] Here is a transcript of a YouTube video.
Please identify the main medical or health related claims made in the video. Use bullet points not numbers.
Transcript:
\"\"\"{revised_text}\"\"\" [/INST]"""

    output = pipe(prompt)[0]["generated_text"]

    st.subheader("Extracted Claims")
    st.write(output)

    claims = output.split("•") if "•" in output else output.split("\n")

    for claim in claims:
        claim = claim.strip()
        if not claim:
            continue
        st.markdown(f"Claim: {claim}")
        sources = search_evidence(claim, Serp_api_key)
        context = "\n".join(sources)

        fact_checking_prompt = f"""<s> [INST] A YouTube video made this medical claim:
\"{claim}\"
Below are the search results from trusted medical sites:
{context}
Based on these sources, assess whether the claim is:
- Medically and/or scientifically accurate
- Potentially misleading
- Uncertain

Then explain why. [/INST]"""

        result = pipe(fact_checking_prompt)[0]["generated_text"]
        st.markdown("Fact Check Result:")
        st.write(result)
        st.divider()
