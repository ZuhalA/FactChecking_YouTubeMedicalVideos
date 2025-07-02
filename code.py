from youtube_transcript_api import YouTubeTranscriptApi
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

login(new_session=False)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text = False
)

# Getting YT videos id, each youtube video has a unique id
yt_url = input("Please Paste YouTube URL Here...")
# This is how a typical yt video url looks like:
# for example: https://www.youtube.com/watch?v=Rn7-ZHHjD8I
# need to split extract the id which is "Rn7-ZHHjD8I"
vid_id = yt_url.split("v=")[-1]
#getting transcript
fetch_ytt_vid = YouTubeTranscriptApi.get_transcript(vid_id, languages = ['en'])


# Cleaning the transcribed text
#since YT transcripts can be in cutted pieces code below makes the transcipt as a long string
fulltext = " ".join(entry['text'] for entry in fetch_ytt_vid) #entry is a loop variable that temporarily holds eahc text
revised_text = re.sub(r'\[.*?\]', '', fulltext) #removing tokens explaining things like, [Laughing]
revised_text = re.sub(r'\s+', ' ', revised_text)
revised_text = revised_text.lower()
print("Sample of Text: \n", revised_text[:1000] + "...")


prompt = f"""<s>[INST] Here is a transcript of a YouTube video.
Please identify the main medical or health related claims made in the video. Use bullet points not numbers.

Transcript:
\"\"\"{revised_text}\"\"\" [/INST]"""

# Generate response
output = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7)[0]["generated_text"]
print("\nExtracted Claims Made in This Video:\n", output)


from serpapi import GoogleSearch


def search_evidence(claim, api_key):
    params = {
        "engine": "google",
        "q": f"{claim} site:mayoclinic.org OR site:cdc.gov OR site:nih.gov OR site:medlineplus.gov",
        "api_key": api_key
    }

    # Perform search and get result as dict
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


Serp_api_key = "YOURS"
claims = output.split("•") if "•" in output else output.split("\n")
for claim in claims:
  claim = claim.strip()
  if not claim:
    continue
  print(f"\n Finding credible evidence for accuracy of this claim: {claim}")
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
  print(result)


