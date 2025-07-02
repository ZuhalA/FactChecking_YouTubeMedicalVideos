import streamlit as st
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from youtube_transcript_api import YouTubeTranscriptApi
from serpapi import GoogleSearch

# Load secrets
HF_TOKEN = st.secrets["HF_API_KEY"]
Serp_api_key = st.secrets["SERP_API_KEY"]

login(token=HF_TOKEN, new_session=False)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# Getting YT videos id, each youtube video has a unique id
yt_url = st.text_input("Please Paste YouTube URL Here...")

if yt_url:
    vid_id = yt_url.split("v=")[-1]

    # getting transcript
    fetch_ytt_vid = YouTubeTranscriptApi.get_transcript(vid_id, languages=['en'])

    # Cleaning the transcribed text
    fulltext = " ".join(entry['text'] for entry in fetch_ytt_vid)
    revised_text = re.sub(r'\[.*?\]', '', fulltext)
    revised_text = re.sub(r'\s+', ' ', revised_text)
    revised_text = revised_text.lower()

    st.write("Sample of Text: \n", revised_text[:1000] + "...")

    prompt = f"""<s>[INST] Here is a transcript of a YouTube video.
Please identify the main medical or health related claims made in the video. Use bullet points not numbers.

Transcript:
\"\"\"{revised_text}\"\"\" [/INST]"""

    output = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7)[0]["generated_text"]

    st.write("\nExtracted Claims Made in This Video:\n", output)

    def search_evidence(claim, api_key):
        params = {
            "engine": "google",
            "q": f"{claim} site:mayoclinic.org OR site:cdc.gov OR site:nih.gov OR site:medlineplus.gov",
            "api_key": api_key
        }

        # Performing search and getting the result as a dict
        search = GoogleSearch(params)
        results = search.get_dict()

        sources = []  # Clean results
        for result in results.get("organic_results", [])[:3]:
            title = result.get("title", "No Title")
            link = result.get("link", "No Link")
            snippet = result.get("snippet", "No Description available.")
            formatted_result = f"Title: {title}\nLink: {link}\nDescription: {snippet}\n"
            sources.append(formatted_result)

        return sources

    claims = output.split("•") if "•" in output else output.split("\n")

    for claim in claims:
        claim = claim.strip()
        if not claim:
            continue
        st.write(f"\nFinding credible evidence for accuracy of this claim: {claim}")
        sources = search_evidence(claim, Serp_api_key)
        context = "\n".join(sources)

        fact_checking_prompt = f""" <s> [INST] A YouTube video made this medical claim:
\"{claim}\"
Below are the search results from trusted medical sites:
{context}
Based on these sources, assess whether the claim is:
- Medically and/or scientifically accurate
- Potentially misleading
- Uncertain

Then explain why. [/INST]
"""
        result = pipe(fact_checking_prompt, max_new_tokens=512)[0]["generated_text"]
        st.write(result)
