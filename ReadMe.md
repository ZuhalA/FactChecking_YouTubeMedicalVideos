# YouTube Medical Misinformation Detection Using LLM & RAG
*A Retrieval-Augmented Generation (RAG) Project Using LLMs*

This project uses a large language model to identify and evaluate **medical or health-related claims** made in YouTube videos, based on evidence from credible sources.

---

## What is RAG?

**Retrieval-Augmented Generation** combines the strengths of language models with real-time external information. Instead of guessing or hallucinating answers, the model first **retrieves facts from trusted websites**, then uses that information to generate grounded and more accurate responses. It's especially useful for tasks like fact-checking, where up-to-date evidence matters.

---

## What it does

- Takes any YouTube link.
- Extracts the full transcript of the video.
- Uses an LLM (Mistral-7B-Instruct) to pull out the major medical or health-related claims.
- Searches trusted sites (like CDC, Mayo Clinic, NIH) for supporting or contradicting evidence.
- Fact-checks each claim using the retrieved content and summarizes whether it’s **accurate**, **uncertain**, or **potentially misleading**.

---

## Process Overview

1. **Transcript Extraction**  
   Captures the full text of the video using `youtube-transcript-api`.

2. **Claim Detection (Generation)**  
   The transcript is passed to an LLM which pulls out individual medical claims.

3. **Evidence Retrieval**  
   Each claim is searched against sites like **mayoclinic.org**, **cdc.gov**, and **nih.gov** using a Google search API.

4. **Evaluation (Grounded Generation)**  
   The LLM reviews each claim in the context of the retrieved evidence and offers a judgment, with reasoning.

---

## Example Output

> **Claim**: *Turmeric has anti-inflammatory properties and can help reduce inflammation in the body.*

> **Assessment**:  
> Based on the search results from trusted medical sites, the claim is **medically and scientifically accurate**. Turmeric contains curcumin, which has been shown to reduce inflammation by inhibiting certain enzymes and inflammatory molecules.  
>  
> Therefore, this claim is **not misleading** and is **well-supported by scientific evidence**. It’s generally considered safe and effective as a supplement to reduce inflammation and support overall health.

---

## Final Note

This is an experimental tool designed to explore how retrieval + generation can be used in real-world misinformation detection. It’s not intended to replace medical advice — always consult with a healthcare professional for reliable health decisions.
