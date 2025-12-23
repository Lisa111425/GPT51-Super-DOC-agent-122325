from typing import List, Dict
import google.generativeai as genai
import anthropic
from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system



def call_llm(

    provider: str,

    model: str,

    system_prompt: str,

    user_messages: List[Dict[str, str]],

    max_tokens: int = 2048,

    temperature: float = 0.4,

) -> str:

    """

    user_messages: list of {"role": "user"/"assistant"/"system", "content": "..."}

    """



    if provider == "OpenAI":

        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

        if not api_key:

            st.error("OpenAI API key is required.")

            return ""

        client = OpenAI(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}] + user_messages

        resp = client.chat.completions.create(

            model=model,

            messages=messages,

            max_tokens=max_tokens,

            temperature=temperature,

        )

        return resp.choices[0].message.content



    elif provider == "Gemini":

        api_key = st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")

        if not api_key:

            st.error("Gemini API key is required.")

            return ""

        genai.configure(api_key=api_key)

        model_obj = genai.GenerativeModel(model)

        full_prompt = f"{system_prompt}\n\n" + "\n\n".join(

            f"{m['role'].upper()}: {m['content']}" for m in user_messages

        )

        resp = model_obj.generate_content(

            full_prompt,

            generation_config=genai.types.GenerationConfig(

                max_output_tokens=max_tokens,

                temperature=temperature,

            ),

        )

        return resp.text



    elif provider == "Anthropic":

        api_key = st.session_state.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:

            st.error("Anthropic API key is required.")

            return ""

        client = anthropic.Anthropic(api_key=api_key)

        # Separate system vs messages explicitly for Anthropic

        messages = [m for m in user_messages if m["role"] != "system"]

        resp = client.messages.create(

            model=model,

            max_tokens=max_tokens,

            temperature=temperature,

            system=system_prompt,

            messages=[{"role": m["role"], "content": m["content"]} for m in messages],

        )

        return "".join(block.text for block in resp.content if hasattr(block, "text"))



    elif provider == "XAI (Grok)":

        # SAMPLE XAI CODE (per your example)

        api_key = st.session_state.get("xai_api_key") or os.getenv("XAI_API_KEY")

        if not api_key:

            st.error("XAI API key is required.")

            return ""

        client = XAIClient(api_key=api_key, timeout=3600)

        chat = client.chat.create(model=model)

        chat.append(xai_system(system_prompt))

        for m in user_messages:

            if m["role"] == "user":

                chat.append(xai_user(m["content"]))

            # For simplicity we ignore assistant history here, or you can add it as needed.



        response = chat.sample()

        return response.content



    else:

        st.error("Unsupported provider.")

        return ""

```



---



## 5. File Handling & New “File Transform & Deep Summary” Agent



### 5.1 File Extraction Helpers



```python

from io import BytesIO

import docx2txt

from PyPDF2 import PdfReader



def extract_text_from_pdf(file_bytes: BytesIO) -> str:

    reader = PdfReader(file_bytes)

    texts = []

    for page in reader.pages:

        texts.append(page.extract_text() or "")

    return "\n".join(texts)



def extract_text_from_docx(file_bytes: BytesIO) -> str:

    return docx2txt.process(file_bytes)



def extract_text_from_txt(file_bytes: BytesIO) -> str:

    return file_bytes.read().decode("utf-8", errors="ignore")



def extract_text(file) -> str:

    name = file.name.lower()

    data = BytesIO(file.read())  # copy into memory

    if name.endswith(".pdf"):

        return extract_text_from_pdf(data)

    elif name.endswith(".docx"):

        return extract_text_from_docx(data)

    elif name.endswith(".txt"):

        return extract_text_from_txt(data)

    else:

        st.error("Unsupported format. Please upload PDF, DOCX, or TXT.")

        return ""

```



### 5.2 Markdown vs PDF Transformation



We’ll treat “Markdown” as LLM-converted structured text; “PDF” as a downloadable PDF built from that markdown.



For PDF we can use a very simple markdown-to-text export plus `reportlab` or `fpdf`:



```python

from fpdf import FPDF



def markdown_to_pdf_bytes(md_text: str) -> bytes:

    pdf = FPDF()

    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()

    pdf.set_font("Arial", size=11)

    for line in md_text.splitlines():

        pdf.multi_cell(0, 5, line)

    pdf_bytes = BytesIO()

    pdf.output(pdf_bytes)

    pdf_bytes.seek(0)

    return pdf_bytes.getvalue()

```



### 5.3 Deep Summary Prompt (2000–3000 words)



```python

def build_deep_summary_prompt(doc_text: str, lang: str) -> str:

    language_instruction = (

        "Write the entire output in English."

        if lang == "en"

        else "請使用繁體中文撰寫整份輸出。"

    )



    base = f"""

You are an elite strategic auditor, legal/financial analyst, and knowledge architect.

{language_instruction}



You are given the full content of a document. Your tasks:



1. Produce a **deep, structured Markdown report** summarizing and analyzing the document.

2. Length: **2,000–3,000 words** (do not be shorter than ~2,000 words unless absolutely necessary).

3. Use the following structure (Markdown headings):



# Executive Overview

- Concise summary of purpose, audience, and domain.



## Key Themes & Objectives

- Bullet list of central goals, decisions, or arguments.



## Detailed Section-by-Section Analysis

- For each major section, provide:

  - What it says

  - Why it matters

  - Risks or opportunities



## Critical Risks, Gaps, and Red Flags

- Legal, compliance, financial, strategic, and operational risks.

- Ambiguities or missing information that might be important.



## Actionable Recommendations

- Concrete steps, prioritized.

- Who should care and why.



## Stakeholder-Specific Views

- How different stakeholders (e.g., executives, auditors, regulators, clients) should interpret the document.



## Glossary of Key Terms (if applicable)

- Short, precise definitions.



Constraints:

- Use Markdown headings (#, ##, ###) and bullet lists.

- Be analytical, not just descriptive.

- When in doubt, **over-explain** key concepts rather than under-explaining.



Now, here is the document content (possibly truncated for length):



[DOCUMENT START]

{doc_text[:100000]}

[DOCUMENT END]

"""

    return base.strip()

```



### 5.4 New Tab: File Transform & Deep Summary



```python

def tab_file_transform_deep_summary():

    st.markdown(f"## {t('tab_file_transform')}")

    st.markdown(

        '<div class="glass-panel">',

        unsafe_allow_html=True,

    )



    uploaded = st.file_uploader(

        t("upload_label"),

        type=["pdf", "docx", "txt"],

        key="file_transform_uploader",

    )



    output_format = st.radio(

        t("output_format"),

        [t("format_markdown"), t("format_pdf")],

        horizontal=True,

        key="output_format_choice",

    )



    if uploaded is not None:

        if st.button(t("run_summary"), type="primary"):

            with st.spinner("Extracting text and generating deep summary…"):

                raw_text = extract_text(uploaded)

                if not raw_text.strip():

                    st.error("No readable text extracted from the file.")

                    return



                provider, model_id, max_tokens, temperature, system_prompt = model_controls()

                lang = st.session_state.get("ui_lang", "en")



                prompt = build_deep_summary_prompt(raw_text, lang)

                output = call_llm(

                    provider=provider,

                    model=model_id,

                    system_prompt=system_prompt,

                    user_messages=[{"role": "user", "content": prompt}],

                    max_tokens=max_tokens,

                    temperature=temperature,

                )

                if not output:

                    return



                # Save to session for RAG-style chat

                st.session_state["latest_file_text"] = raw_text

                st.session_state["latest_file_summary_md"] = output

                st.session_state["latest_file_name"] = uploaded.name



                # Show markdown summary

                st.markdown("### 📄 Deep Summary (Markdown)")

                st.markdown(output)



                # Download options

                if output_format == t("format_markdown"):

                    st.download_button(

                        "Download Markdown",

                        data=output.encode("utf-8"),

                        file_name=f"{uploaded.name}.summary.md",

                        mime="text/markdown",

                    )

                else:

                    pdf_bytes = markdown_to_pdf_bytes(output)

                    st.download_button(

                        "Download PDF",

                        data=pdf_bytes,

                        file_name=f"{uploaded.name}.summary.pdf",

                        mime="application/pdf",

                    )



    # Chat with file

    if "latest_file_text" in st.session_state:

        st.markdown("---")

        st.markdown(f"### 💬 {t('chat_with_file')} — {st.session_state.get('latest_file_name', '')}")

        user_q = st.text_area(t("user_prompt"), key="file_chat_prompt")

        if st.button("Ask the file"):

            provider, model_id, max_tokens, temperature, system_prompt = model_controls()

            full_context = f"""

You are given a source document and a long-form summary of that document.



[ORIGINAL DOCUMENT]

{st.session_state['latest_file_text'][:60000]}



[SUMMARY]

{st.session_state['latest_file_summary_md'][:40000]}



Answer the user's question strictly grounded in these sources.

If the answer is not supported by the information above, say so clearly.

"""

            question = user_q.strip()

            if not question:

                st.warning("Please enter a question.")

                return



            with st.spinner("Thinking with the document…"):

                answer = call_llm(

                    provider=provider,

                    model=model_id,

                    system_prompt=system_prompt,

                    user_messages=[

                        {"role": "user", "content": full_context},

                        {"role": "user", "content": question},

                    ],

                    max_tokens=max_tokens,

                    temperature=temperature,

                )

            st.markdown("#### Answer")

            st.markdown(answer or "_No answer produced._")



    st.markdown("</div>", unsafe_allow_html=True)

```



---



## 6. Other Tabs (Kept Features, Simplified)



Below are minimal but functional skeletons for the 4 original modules, using the same `call_llm` and controls. You can extend prompts as deeply as you like.



### 6.1 File Intelligence (Single File)



```python

def tab_file_intelligence():

    st.markdown(f"## {t('tab_file_intel')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    up = st.file_uploader(

        t("upload_label"),

        type=["pdf", "docx", "txt", "md"],

        key="file_intel_uploader",

    )

    if up is not None and st.button("Analyze File"):

        with st.spinner("Analyzing file…"):

            text = extract_text(up)

            provider, model_id, max_tokens, temperature, system_prompt = model_controls()

            lang = st.session_state.get("ui_lang", "en")



            language_instruction = (

                "Write the output in English."

                if lang == "en"

                else "請使用繁體中文撰寫輸出。"

            )

            prompt = f"""

You are an elite strategic auditor and document analyst.

{language_instruction}



Analyze the following document and produce a structured Markdown report.



[DOCUMENT START]

{text[:100000]}

[DOCUMENT END]

"""

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

            st.markdown("### Analysis")

            st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)

```



### 6.2 Multi-File Synthesis



```python

def tab_multi_file_synthesis():

    st.markdown(f"## {t('tab_multi_file')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    files = st.file_uploader(

        "Upload multiple files (PDF/DOCX/TXT/MD)",

        type=["pdf", "docx", "txt", "md"],

        accept_multiple_files=True,

        key="multi_files",

    )

    if files and st.button("Combine & Analyze"):

        with st.spinner("Combining and analyzing files…"):

            assembled = []

            for f in files:

                content = extract_text(f)

                assembled.append(f"--- START FILE: {f.name} ---\n{content}\n--- END FILE: {f.name} ---\n")

            combined = "\n".join(assembled)[:150000]



            provider, model_id, max_tokens, temperature, system_prompt = model_controls()

            lang = st.session_state.get("ui_lang", "en")

            language_instruction = (

                "Write the output in English."

                if lang == "en"

                else "請使用繁體中文撰寫輸出。"

            )



            prompt = f"""

You are an advanced synthesis engine (strategic auditor / consultant).

{language_instruction}



You are given multiple documents enclosed in START/END FILE markers.

Treat them as a unified knowledge base.



Tasks:

- Compare and contrast documents.

- Identify trends, inconsistencies, and deltas across files.

- Provide a structured Markdown report with:

  - Executive Summary

  - Cross-Document Comparisons

  - Key Risks / Gaps

  - Recommended Next Actions.



[DOCUMENTS]

{combined}

"""

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

            st.markdown("### Synthesis Report")

            st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)

```



### 6.3 Smart Replace (Template Filling)



```python

def tab_smart_replace():

    st.markdown(f"## {t('tab_smart_replace')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        template_text = st.text_area("Template (with placeholders like [Client Name])", height=260)

    with col2:

        context_text = st.text_area("Context / Raw Data Source", height=260)



    instructions = st.text_area(

        "Natural language instructions (tone, style, constraints)",

        value="Fill in all placeholders using the context. Keep it legally precise and professional.",

    )



    if st.button("Run Smart Replace"):

        provider, model_id, max_tokens, temperature, system_prompt = model_controls()

        lang = st.session_state.get("ui_lang", "en")

        language_instruction = (

            "Write the output in English."

            if lang == "en"

            else "請使用繁體中文撰寫輸出。"

        )



        prompt = f"""

You are a legal and business drafting expert.



{language_instruction}



Here is a template document with placeholders like [Client Name], [Effective Date], etc.



[TEMPLATE]

{template_text}



Here is unstructured context that contains the values you should infer:

[CONTEXT]

{context_text}



Instructions from the user:

{instructions}



Tasks:

- Infer values for all placeholders from CONTEXT.

- Replace placeholders carefully and adjust surrounding text so it reads naturally.

- Do not invent facts not reasonably supported by the context.

- Output only the **completed template** in Markdown.

"""

        with st.spinner("Generating filled template…"):

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

        st.markdown("### Completed Template")

        st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)

```



### 6.4 AI Note Keeper



```python

def tab_ai_note_keeper():

    st.markdown(f"## {t('tab_note_keeper')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)



    raw_note = st.text_area("Your raw notes / brain dump", height=240, key="note_raw")

    col1, col2, col3, col4, col5 = st.columns(5)

    action = None

    if col1.button("Format"):

        action = "format"

    if col2.button("Tasks"):

        action = "tasks"

    if col3.button("Fix"):

        action = "fix"

    if col4.button("Summary"):

        action = "summary"

    if col5.button("Expand"):

        action = "expand"



    if action and raw_note.strip():

        provider, model_id, max_tokens, temperature, system_prompt = model_controls()

        lang = st.session_state.get("ui_lang", "en")

        language_instruction = (

            "Write the output in English."

            if lang == "en"

            else "請使用繁體中文撰寫輸出。"

        )



        prompt_map = {

            "format": "Reformat these notes into clean Markdown with clear headings and bullet points.",

            "tasks": "Extract all actionable tasks and output them as a checklist (- [ ] ...).",

            "fix": "Fix grammar, spelling, and clarity, keeping a business-professional tone.",

            "summary": "Write a concise TL;DR summary followed by a structured outline.",

            "expand": "Expand terse bullets into fleshed-out paragraphs with examples.",

        }

        prompt = f"""

You are an expert personal knowledge manager.



{language_instruction}



User's raw notes:

{raw_note}



Task: {prompt_map[action]}



Output only the transformed notes in Markdown.

"""

        with st.spinner("Transforming notes…"):

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

        st.markdown("### Transformed Notes")

        st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)

```



---



## 7. Putting It All Together: `app.py` Skeleton



```python

import streamlit as st



# (import everything above…)



def main():

    # Initialize session defaults

    st.set_page_config(

        page_title="AuditFlow AI · Masterpiece Edition",

        layout="wide",

    )

    if "ui_lang" not in st.session_state:

        st.session_state.ui_lang = "en"

    if "dark_mode" not in st.session_state:

        st.session_state.dark_mode = True

    if "artist_style_key" not in st.session_state:

        st.session_state.artist_style_key = "van_gogh"



    # Sidebar: global controls

    with st.sidebar:

        st.markdown("## 🎨 Masterpiece Controls")

        lang = st.radio("Language / 語言", ["English", "繁體中文"], key="lang_radio")

        st.session_state.ui_lang = "en" if lang == "English" else "zh"



        dark_mode = st.toggle("Dark mode", value=True, key="dark_mode_toggle")

        st.session_state.dark_mode = dark_mode



        active_style = style_selector_ui()

        provider = api_keys_section()

        provider, model_id, max_tokens, temperature, system_prompt = model_controls()



    # Apply theme

    apply_theme(active_style, st.session_state.dark_mode)



    # Main header

    st.markdown(f"<h1 class='accent-title'>{t('app_title')}</h1>", unsafe_allow_html=True)

    st.markdown(t("subtitle"))



    # Tabs

    tab1, tab2, tab3, tab4, tab5 = st.tabs([

        t("tab_file_transform"),

        t("tab_file_intel"),

        t("tab_multi_file"),

        t("tab_smart_replace"),

        t("tab_note_keeper"),

    ])



    with tab1:

        tab_file_transform_deep_summary()

    with tab2:

        tab_file_intelligence()

    with tab3:

        tab_multi_file_synthesis()

    with tab4:

        tab_smart_replace()

    with tab5:

        tab_ai_note_keeper()



if __name__ == "__main__":

    main()

```



This gives you:



- Agentic flows (file ingest → analysis → RAG Q&A).

- Wow-style painter UI with light/dark, EN/繁中, and jackpot styling.

- User control over model, prompt, tokens before running agents.

- BYOK model with safe env-key usage (no key display if from env).

- Sample Grok (XAI) integration built exactly on your sample API snippet concept.



---



## 8. 20 Follow-Up Questions



1. For the **2000–3000 word summary**, do you want a **hard length constraint** (model forced to reach that size) or a more flexible “aim for that range” as it is now?

2. Should the **output language** always match the **UI language**, or do you want a **separate toggle** for “UI language” vs “analysis output language”?

3. In the new **File Transform & Deep Summary** tab, do you also want the option to **download the original extracted plain text** alongside the markdown/PDF?

4. For **RAG-style chat**, do you prefer a **simple full-context injection** (current approach) or should we integrate an actual **chunking + similarity search** pipeline (e.g., local embeddings) for larger documents?

5. Do you want the **chat history with each file** to be persisted per session (multi-turn context) or should each question be treated as **stateless**?

6. For **Multi-File Synthesis**, should users be able to **tag files by category** (e.g., “Q1 report”, “Contract”, “Policy”) and have the model reason using those tags explicitly?

7. Would you like an explicit **"Risk Heatmap" section** (e.g., Low/Medium/High with colored chips) in the summary outputs, which we can enforce via prompt instructions?

8. For the **Smart Replace** module, do you need **export to DOCX/PDF** in addition to markdown, or is markdown sufficient for now?

9. Should **AI Note Keeper** support **multiple saved notes** with titles (mini knowledge base) or is a single working area per session sufficient?

10. Do you want a specific **default model mapping per tab** (e.g., Gemini 3 Flash for synthesis, Grok for creative reasoning, etc.) or keep a single global model selection?

11. For **API usage safety**, would you like a simple **token/usage estimator** shown before each run to help users understand potential cost?

12. Are there any **compliance or PII constraints** (e.g., never send certain fields to the LLM) that require **in-browser redaction** before the text goes to the API?

13. Should the **“Jackpot” style** also randomize between **light/dark** or only shuffle painter styles while preserving the current theme?

14. Do you want **per-style overrides** of font sizes and spacing (e.g., more compact for Banksy, more airy for Monet) or keep layout consistent for readability?

15. For **Traditional Chinese**, do you prefer a **Taiwan-style vocabulary and punctuation** or a more generic Traditional Chinese style?

16. Would you like an optional **“auditor persona selection”** (legal / financial / strategic / technical) that changes the system prompt templates and the type of analysis?

17. Is support for **image-containing PDFs** (OCR) important for your first version, or can we assume text-based PDFs only for now?

18. Should we add a **“raw LLM playground” tab** where users can freely edit system prompt, messages, model, and tokens without any document context?

19. Do you want a **simple telemetry panel** (client-side only) that counts how many calls were made per provider/model in the current session?

20. Are there any **specific famous painters or visual brands** you want added or replaced in the current list of 20 styles to better match your product identity?

ファイルが選択されていませんファイルが選択されていません
Super, please create app.py, advanced agents.yaml with 31 agents that is handy for FDA in traditional chinese. Ending with 20 comprehensive follow up questions.

```python

# app.py

# AuditFlow AI · Masterpiece Edition (FDA-focused, Agentic, Painter-Themed)

# Designed for Streamlit on Hugging Face Spaces

#

# Features:

# - Wow UI with 20 painter styles + Jackpot selector

# - Light/Dark theme

# - English / 繁體中文 UI

# - Agent selection from advanced_agents.yaml (31 FDA-oriented agents, in Traditional Chinese)

# - BYOK for OpenAI / Gemini / Anthropic / XAI (Grok), with env fallback (hidden)

# - File Transform & Deep Summary tab (PDF/DOCX/TXT → Markdown/PDF + 2,000–3,000 word summary)

# - File Intelligence, Multi-File Synthesis, Smart Replace, AI Note Keeper

# - Model selection, max tokens, temperature, customizable system prompt before executing agents



import os

import time

import random

from dataclasses import dataclass

from io import BytesIO

from typing import List, Dict, Any



import streamlit as st

import yaml

from openai import OpenAI

import google.generativeai as genai

import anthropic

from xai_sdk import Client as XAIClient

from xai_sdk.chat import user as xai_user, system as xai_system



import docx2txt

from PyPDF2 import PdfReader

from fpdf import FPDF





# =========================

#  Localization

# =========================



UI_TEXT = {

    "en": {

        "app_title": "AuditFlow AI · Masterpiece Edition (FDA)",

        "subtitle": "FDA-oriented agentic document intelligence with painterly themes.",

        "tab_file_transform": "File Transform & Deep Summary",

        "tab_file_intel": "File Intelligence",

        "tab_multi_file": "Multi-File Synthesis",

        "tab_smart_replace": "Smart Replace",

        "tab_note_keeper": "AI Note Keeper",

        "upload_label": "Upload a document (PDF, DOCX, TXT):",

        "output_format": "Transform file into:",

        "format_markdown": "Markdown (.md)",

        "format_pdf": "PDF (.pdf)",

        "run_summary": "Generate 2,000–3,000 word Masterpiece summary",

        "chat_with_file": "Chat with this file",

        "api_key_section": "API Keys (browser-only, never sent to any server except LLM provider)",

        "provider": "Provider",

        "model": "Model",

        "custom_prompt": "Custom system prompt",

        "max_tokens": "Max tokens",

        "temperature": "Temperature",

        "user_prompt": "Your question / instruction",

        "agent_select": "FDA Agent (from advanced_agents.yaml)",

    },

    "zh": {

        "app_title": "AuditFlow AI · 大師傑作版（FDA 專用）",

        "subtitle": "面向 FDA 報規與合規需求的代理式文件智慧系統，結合藝術風格體驗。",

        "tab_file_transform": "檔案轉換與深度摘要",

        "tab_file_intel": "單一文件分析",

        "tab_multi_file": "多文件綜合分析",

        "tab_smart_replace": "智慧範本填寫",

        "tab_note_keeper": "AI 筆記管理員",

        "upload_label": "上傳文件（PDF、DOCX、TXT）：",

        "output_format": "將檔案轉換為：",

        "format_markdown": "Markdown (.md)",

        "format_pdf": "PDF (.pdf)",

        "run_summary": "產生 2,000–3,000 字深度摘要（Markdown）",

        "chat_with_file": "針對此文件發問",

        "api_key_section": "API 金鑰（僅在本機瀏覽器中使用，僅送往 LLM 供應商）",

        "provider": "服務提供者",

        "model": "模型",

        "custom_prompt": "自訂系統提示（System Prompt）",

        "max_tokens": "最大 Token 數",

        "temperature": "溫度",

        "user_prompt": "你的問題 / 指令",

        "agent_select": "FDA 代理人（來自 advanced_agents.yaml）",

    },

}





def t(key: str) -> str:

    lang = st.session_state.get("ui_lang", "en")

    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)





# =========================

#  Painter Styles

# =========================



@dataclass

class ArtistStyle:

    key: str

    display_name: str

    painter: str

    bg_gradient_light: str

    bg_gradient_dark: str

    panel_bg_rgba: str

    accent_color: str

    accent_soft: str

    font_family: str





ARTIST_STYLES: List[ArtistStyle] = [

    ArtistStyle(

        key="van_gogh",

        display_name="Starry Night",

        painter="Vincent van Gogh",

        bg_gradient_light="linear-gradient(135deg,#fdfbfb 0%,#ebedee 100%)",

        bg_gradient_dark="linear-gradient(135deg,#0f172a 0%,#1e293b 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.75)",

        accent_color="#facc15",

        accent_soft="#fef9c3",

        font_family="'DM Sans', system-ui, -apple-system, BlinkMacSystemFont, sans-serif",

    ),

    ArtistStyle(

        key="monet",

        display_name="Water Lilies",

        painter="Claude Monet",

        bg_gradient_light="linear-gradient(135deg,#e0f4ff 0%,#f9f7ff 100%)",

        bg_gradient_dark="linear-gradient(135deg,#0b1120 0%,#1d2233 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.70)",

        accent_color="#22c55e",

        accent_soft="#dcfce7",

        font_family="'Playfair Display', Georgia, 'Times New Roman', serif",

    ),

    ArtistStyle(

        key="picasso",

        display_name="Cubist Geometry",

        painter="Pablo Picasso",

        bg_gradient_light="linear-gradient(135deg,#fdfbfb 0%,#ebedee 40%,#fee2e2 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#111827 50%,#1f2933 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.80)",

        accent_color="#f97316",

        accent_soft="#ffedd5",

        font_family="'Space Grotesk', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="da_vinci",

        display_name="Renaissance Studio",

        painter="Leonardo da Vinci",

        bg_gradient_light="linear-gradient(135deg,#faf5e4 0%,#fef9c3 100%)",

        bg_gradient_dark="linear-gradient(135deg,#1c1917 0%,#292524 100%)",

        panel_bg_rgba="rgba(24, 24, 27, 0.85)",

        accent_color="#fbbf24",

        accent_soft="#fef3c7",

        font_family="'Crimson Text', Georgia, 'Times New Roman', serif",

    ),

    ArtistStyle(

        key="michelangelo",

        display_name="Sistine Ceiling",

        painter="Michelangelo",

        bg_gradient_light="linear-gradient(135deg,#e5e7eb 0%,#f9fafb 100%)",

        bg_gradient_dark="linear-gradient(135deg,#111827 0%,#020617 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.78)",

        accent_color="#60a5fa",

        accent_soft="#dbeafe",

        font_family="'Playfair Display', Georgia, serif",

    ),

    ArtistStyle(

        key="rembrandt",

        display_name="Chiaroscuro",

        painter="Rembrandt",

        bg_gradient_light="linear-gradient(135deg,#fef3c7 0%,#fed7aa 100%)",

        bg_gradient_dark="linear-gradient(135deg,#0b1120 0%,#1f2937 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.9)",

        accent_color="#f97316",

        accent_soft="#ffedd5",

        font_family="'Merriweather', Georgia, serif",

    ),

    ArtistStyle(

        key="klimt",

        display_name="Golden Mosaic",

        painter="Gustav Klimt",

        bg_gradient_light="linear-gradient(135deg,#fef3c7 0%,#facc15 50%,#fee2e2 100%)",

        bg_gradient_dark="linear-gradient(135deg,#1c1917 0%,#3f3f46 100%)",

        panel_bg_rgba="rgba(24, 24, 27, 0.85)",

        accent_color="#eab308",

        accent_soft="#fef9c3",

        font_family="'Cormorant Garamond', Georgia, serif",

    ),

    ArtistStyle(

        key="matisse",

        display_name="Cut-Outs",

        painter="Henri Matisse",

        bg_gradient_light="linear-gradient(135deg,#fee2e2 0%,#f9a8d4 50%,#bfdbfe 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#0f172a 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.9)",

        accent_color="#ec4899",

        accent_soft="#fce7f3",

        font_family="'Fredoka', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="dali",

        display_name="Surreal Desert",

        painter="Salvador Dalí",

        bg_gradient_light="linear-gradient(135deg,#fef3c7 0%,#fde68a 40%,#bfdbfe 100%)",

        bg_gradient_dark="linear-gradient(135deg,#111827 0%,#1f2937 100%)",

        panel_bg_rgba="rgba(17, 24, 39, 0.85)",

        accent_color="#f97316",

        accent_soft="#ffedd5",

        font_family="'IBM Plex Sans', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="warhol",

        display_name="Pop Factory",

        painter="Andy Warhol",

        bg_gradient_light="linear-gradient(135deg,#f9a8d4 0%,#f97316 40%,#22c55e 70%,#38bdf8 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#111827 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.9)",

        accent_color="#ec4899",

        accent_soft="#fee2e2",

        font_family="'Poppins', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="hokusai",

        display_name="Great Wave",

        painter="Hokusai",

        bg_gradient_light="linear-gradient(135deg,#e0f2fe 0%,#bfdbfe 40%,#f1f5f9 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#0f172a 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.9)",

        accent_color="#0ea5e9",

        accent_soft="#dbeafe",

        font_family="'Noto Sans TC', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="frida",

        display_name="Vivid Blossoms",

        painter="Frida Kahlo",

        bg_gradient_light="linear-gradient(135deg,#fee2e2 0%,#fecaca 40%,#bbf7d0 100%)",

        bg_gradient_dark="linear-gradient(135deg,#0f172a 0%,#1e293b 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.85)",

        accent_color="#f97316",

        accent_soft="#ffedd5",

        font_family="'Josefin Sans', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="banksy",

        display_name="Street Brutalism",

        painter="Banksy",

        bg_gradient_light="linear-gradient(135deg,#e5e7eb 0%,#9ca3af 40%,#111827 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#111827 100%)",

        panel_bg_rgba="rgba(17, 24, 39, 0.95)",

        accent_color="#f97316",

        accent_soft="#fee2e2",

        font_family="'Roboto Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, monospace",

    ),

    ArtistStyle(

        key="rothko",

        display_name="Color Fields",

        painter="Mark Rothko",

        bg_gradient_light="linear-gradient(135deg,#fecaca 0%,#fed7aa 40%,#fef9c3 100%)",

        bg_gradient_dark="linear-gradient(135deg,#111827 0%,#1f2937 100%)",

        panel_bg_rgba="rgba(17, 24, 39, 0.9)",

        accent_color="#fb7185",

        accent_soft="#fee2e2",

        font_family="'Work Sans', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="chagall",

        display_name="Dreamscapes",

        painter="Marc Chagall",

        bg_gradient_light="linear-gradient(135deg,#e0e7ff 0%,#f5d0fe 40%,#cffafe 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#0f172a 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.9)",

        accent_color="#a855f7",

        accent_soft="#ede9fe",

        font_family="'Quicksand', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="basquiat",

        display_name="Neo-Expressionism",

        painter="Jean-Michel Basquiat",

        bg_gradient_light="linear-gradient(135deg,#fee2e2 0%,#f97316 30%,#22c55e 60%,#38bdf8 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#0f172a 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.95)",

        accent_color="#facc15",

        accent_soft="#fef3c7",

        font_family="'Inter', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="turner",

        display_name="Storm Light",

        painter="J. M. W. Turner",

        bg_gradient_light="linear-gradient(135deg,#fef3c7 0%,#fde68a 50%,#bfdbfe 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#111827 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.85)",

        accent_color="#f59e0b",

        accent_soft="#fef3c7",

        font_family="'DM Serif Display', Georgia, serif",

    ),

    ArtistStyle(

        key="vermeer",

        display_name="Soft Interior",

        painter="Johannes Vermeer",

        bg_gradient_light="linear-gradient(135deg,#e5e7eb 0%,#e0f2fe 50%,#fef9c3 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#0f172a 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.85)",

        accent_color="#38bdf8",

        accent_soft="#dbeafe",

        font_family="'Lora', Georgia, serif",

    ),

    ArtistStyle(

        key="cezanne",

        display_name="Mountain Geometry",

        painter="Paul Cézanne",

        bg_gradient_light="linear-gradient(135deg,#e0f2fe 0%,#bbf7d0 50%,#fee2e2 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#0f172a 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.9)",

        accent_color="#22c55e",

        accent_soft="#dcfce7",

        font_family="'Source Sans 3', system-ui, sans-serif",

    ),

    ArtistStyle(

        key="pollock",

        display_name="Action Painting",

        painter="Jackson Pollock",

        bg_gradient_light="linear-gradient(135deg,#f1f5f9 0%,#e5e7eb 30%,#fecaca 60%,#bef264 100%)",

        bg_gradient_dark="linear-gradient(135deg,#020617 0%,#111827 100%)",

        panel_bg_rgba="rgba(15, 23, 42, 0.95)",

        accent_color="#f97316",

        accent_soft="#fed7aa",

        font_family="'Manrope', system-ui, sans-serif",

    ),

]





def apply_theme(style: ArtistStyle, dark_mode: bool):

    bg = style.bg_gradient_dark if dark_mode else style.bg_gradient_light

    panel = style.panel_bg_rgba

    text_color = "#e5e7eb" if dark_mode else "#020617"



    css = f"""

    <style>

    html, body, [data-testid="stAppViewContainer"] {{

        background: {bg} !important;

        background-attachment: fixed;

        font-family: {style.font_family};

        color: {text_color};

    }}

    .glass-panel {{

        background: {panel};

        backdrop-filter: blur(18px);

        -webkit-backdrop-filter: blur(18px);

        border-radius: 20px;

        border: 1px solid rgba(255,255,255,0.18);

        padding: 1.25rem 1.5rem;

        margin-bottom: 1.5rem;

    }}

    .accent-title {{

        color: {style.accent_color};

    }}

    .accent-chip {{

        background: {style.accent_soft};

        color: #111827;

        border-radius: 9999px;

        padding: 0.15rem 0.7rem;

        font-size: 0.75rem;

        font-weight: 500;

        display: inline-flex;

        align-items: center;

        gap: 0.25rem;

    }}

    textarea, .stTextInput > div > div > input {{

        background: rgba(15,23,42,0.75) !important;

        color: #e5e7eb !important;

    }}

    </style>

    """

    st.markdown(css, unsafe_allow_html=True)





def style_selector_ui() -> ArtistStyle:

    st.markdown("### 🎨 Masterpiece Style Jackpot")

    style_keys = [s.key for s in ARTIST_STYLES]

    current_style_key = st.session_state.get("artist_style_key", "van_gogh")



    col1, col2 = st.columns([3, 1])

    with col1:

        selected_key = st.selectbox(

            "Style",

            options=style_keys,

            index=style_keys.index(current_style_key) if current_style_key in style_keys else 0,

            format_func=lambda k: next(s.display_name for s in ARTIST_STYLES if s.key == k),

            key="artist_style_dropdown",

        )

    with col2:

        if st.button("Inspire Me (Jackpot)"):

            placeholder = st.empty()

            for _ in range(15):

                rand_key = random.choice(style_keys)

                st.session_state.artist_style_key = rand_key

                placeholder.write(

                    f"🎰 🎨 {next(s.display_name for s in ARTIST_STYLES if s.key == rand_key)}"

                )

                time.sleep(0.06)

            placeholder.empty()



    st.session_state.artist_style_key = st.session_state.get("artist_style_key", selected_key)

    active_style = next(s for s in ARTIST_STYLES if s.key == st.session_state.artist_style_key)

    return active_style





# =========================

#  Agents (from YAML)

# =========================



def load_agents(path: str = "advanced_agents.yaml") -> List[Dict[str, Any]]:

    try:

        with open(path, "r", encoding="utf-8") as f:

            data = yaml.safe_load(f)

        return data.get("agents", [])

    except Exception as e:

        st.sidebar.error(f"Failed to load agents YAML: {e}")

        return []





def agent_selector_ui(agents: List[Dict[str, Any]]) -> Dict[str, Any]:

    if not agents:

        st.sidebar.warning("No agents loaded from advanced_agents.yaml.")

        return {}



    st.sidebar.markdown(f"### 🤖 {t('agent_select')}")

    ids = [a["id"] for a in agents]



    def label_func(agent_id: str) -> str:

        a = next(ag for ag in agents if ag["id"] == agent_id)

        return a.get("display_name_zh", agent_id)



    default_idx = 0

    if "selected_agent_id" in st.session_state:

        try:

            default_idx = ids.index(st.session_state["selected_agent_id"])

        except ValueError:

            default_idx = 0



    selected_id = st.sidebar.selectbox(

        "Agent",

        options=ids,

        index=default_idx,

        format_func=label_func,

        key="agent_selectbox",

    )

    selected_agent = next(a for a in agents if a["id"] == selected_id)



    # When agent changes, overwrite model & prompt defaults

    if st.session_state.get("selected_agent_id") != selected_id:

        st.session_state["selected_agent_id"] = selected_id

        # Defaults from agent config

        st.session_state["llm_provider"] = selected_agent.get("default_provider", "Gemini")

        st.session_state["llm_model_id"] = selected_agent.get("default_model", "gemini-3-flash")

        st.session_state["llm_max_tokens"] = selected_agent.get("default_max_tokens", 4096)

        st.session_state["llm_temperature"] = selected_agent.get("default_temperature", 0.3)

        st.session_state["llm_system_prompt"] = selected_agent.get(

            "system_prompt_zh",

            "你是一位 FDA 法規合規與策略分析專家，請使用繁體中文回答。",

        )



    return selected_agent





# =========================

#  API Keys

# =========================



def get_env_or_session_api_key(session_key: str, env_var: str) -> str:

    if session_key in st.session_state and st.session_state[session_key]:

        return st.session_state[session_key]



    env_val = os.getenv(env_var)

    if env_val:

        # Use environment key silently

        st.session_state[session_key] = env_val

        return env_val

    return ""





def render_api_key_inputs():

    st.sidebar.markdown(f"### 🔐 {t('api_key_section')}")

    with st.sidebar.expander("OpenAI", expanded=False):

        env_val = os.getenv("OPENAI_API_KEY")

        if env_val:

            st.markdown("Using environment OpenAI API key（不顯示實際值）。")

            st.session_state["openai_api_key"] = env_val

        else:

            st.session_state["openai_api_key"] = st.text_input(

                "OpenAI API Key",

                type="password",

                value=st.session_state.get("openai_api_key", ""),

            )



    with st.sidebar.expander("Gemini", expanded=False):

        env_val = os.getenv("GEMINI_API_KEY")

        if env_val:

            st.markdown("Using environment Gemini API key（不顯示實際值）。")

            st.session_state["gemini_api_key"] = env_val

        else:

            st.session_state["gemini_api_key"] = st.text_input(

                "Gemini API Key",

                type="password",

                value=st.session_state.get("gemini_api_key", ""),

            )



    with st.sidebar.expander("Anthropic", expanded=False):

        env_val = os.getenv("ANTHROPIC_API_KEY")

        if env_val:

            st.markdown("Using environment Anthropic API key（不顯示實際值）。")

            st.session_state["anthropic_api_key"] = env_val

        else:

            st.session_state["anthropic_api_key"] = st.text_input(

                "Anthropic API Key",

                type="password",

                value=st.session_state.get("anthropic_api_key", ""),

            )



    with st.sidebar.expander("XAI (Grok)", expanded=False):

        env_val = os.getenv("XAI_API_KEY")

        if env_val:

            st.markdown("Using environment XAI API key（不顯示實際值）。")

            st.session_state["xai_api_key"] = env_val

        else:

            st.session_state["xai_api_key"] = st.text_input(

                "XAI API Key",

                type="password",

                value=st.session_state.get("xai_api_key", ""),

            )





# =========================

#  Model & Prompt Controls

# =========================



MODEL_CATALOG = {

    "OpenAI": [

        {"id": "gpt-4o-mini", "label": "GPT‑4o mini"},

        {"id": "gpt-4.1-mini", "label": "GPT‑4.1 mini"},

    ],

    "Gemini": [

        {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},

        {"id": "gemini-3-flash", "label": "Gemini 3 Flash"},

    ],

    "Anthropic": [

        {"id": "claude-3.5-sonnet", "label": "Claude 3.5 Sonnet"},

        {"id": "claude-3.5-haiku", "label": "Claude 3.5 Haiku"},

    ],

    "XAI (Grok)": [

        {"id": "grok-4", "label": "Grok-4 (XAI)"},

    ],

}





def render_llm_controls():

    st.sidebar.markdown("### 🧠 LLM & Prompt")

    provider = st.sidebar.selectbox(

        t("provider"),

        list(MODEL_CATALOG.keys()),

        index=list(MODEL_CATALOG.keys()).index(st.session_state.get("llm_provider", "Gemini")),

        key="llm_provider",

    )

    models = MODEL_CATALOG[provider]

    model_ids = [m["id"] for m in models]



    default_model = st.session_state.get("llm_model_id", model_ids[0])

    if default_model not in model_ids:

        default_model = model_ids[0]



    model_id = st.sidebar.selectbox(

        t("model"),

        options=model_ids,

        index=model_ids.index(default_model),

        format_func=lambda m: next(x["label"] for x in models if x["id"] == m),

        key="llm_model_id",

    )



    max_tokens = st.sidebar.slider(

        t("max_tokens"), min_value=256, max_value=8192, value=int(st.session_state.get("llm_max_tokens", 4096)), step=256,

        key="llm_max_tokens",

    )

    temperature = st.sidebar.slider(

        t("temperature"),

        min_value=0.0,

        max_value=1.5,

        value=float(st.session_state.get("llm_temperature", 0.3)),

        step=0.05,

        key="llm_temperature",

    )

    system_prompt = st.sidebar.text_area(

        t("custom_prompt"),

        value=st.session_state.get("llm_system_prompt", ""),

        key="llm_system_prompt",

        height=180,

    )

    return provider, model_id, max_tokens, temperature, system_prompt





def get_llm_config():

    return (

        st.session_state.get("llm_provider", "Gemini"),

        st.session_state.get("llm_model_id", "gemini-3-flash"),

        int(st.session_state.get("llm_max_tokens", 4096)),

        float(st.session_state.get("llm_temperature", 0.3)),

        st.session_state.get("llm_system_prompt", "你是一位 FDA 法規合規與策略分析專家，請使用繁體中文回答。"),

    )





# =========================

#  LLM Call Wrapper

# =========================



def call_llm(

    provider: str,

    model: str,

    system_prompt: str,

    user_messages: List[Dict[str, str]],

    max_tokens: int = 2048,

    temperature: float = 0.4,

) -> str:

    if provider == "OpenAI":

        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

        if not api_key:

            st.error("OpenAI API key is required.")

            return ""

        client = OpenAI(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}] + user_messages

        resp = client.chat.completions.create(

            model=model,

            messages=messages,

            max_tokens=max_tokens,

            temperature=temperature,

        )

        return resp.choices[0].message.content



    elif provider == "Gemini":

        api_key = st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")

        if not api_key:

            st.error("Gemini API key is required.")

            return ""

        genai.configure(api_key=api_key)

        model_obj = genai.GenerativeModel(model)

        full_prompt = f"{system_prompt}\n\n" + "\n\n".join(

            f"{m['role'].upper()}: {m['content']}" for m in user_messages

        )

        resp = model_obj.generate_content(

            full_prompt,

            generation_config=genai.types.GenerationConfig(

                max_output_tokens=max_tokens,

                temperature=temperature,

            ),

        )

        return resp.text



    elif provider == "Anthropic":

        api_key = st.session_state.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:

            st.error("Anthropic API key is required.")

            return ""

        client = anthropic.Anthropic(api_key=api_key)

        messages = [m for m in user_messages if m["role"] != "system"]

        resp = client.messages.create(

            model=model,

            max_tokens=max_tokens,

            temperature=temperature,

            system=system_prompt,

            messages=[{"role": m["role"], "content": m["content"]} for m in messages],

        )

        return "".join(block.text for block in resp.content if hasattr(block, "text"))



    elif provider == "XAI (Grok)":

        # Sample XAI usage, following your reference

        api_key = st.session_state.get("xai_api_key") or os.getenv("XAI_API_KEY")

        if not api_key:

            st.error("XAI API key is required.")

            return ""

        client = XAIClient(api_key=api_key, timeout=3600)

        chat = client.chat.create(model=model)

        chat.append(xai_system(system_prompt))

        for m in user_messages:

            if m["role"] == "user":

                chat.append(xai_user(m["content"]))

        response = chat.sample()

        return response.content



    else:

        st.error("Unsupported provider.")

        return ""





# =========================

#  File Utilities

# =========================



def extract_text_from_pdf(file_bytes: BytesIO) -> str:

    reader = PdfReader(file_bytes)

    texts = []

    for page in reader.pages:

        texts.append(page.extract_text() or "")

    return "\n".join(texts)





def extract_text_from_docx(file_bytes: BytesIO) -> str:

    return docx2txt.process(file_bytes)





def extract_text_from_txt(file_bytes: BytesIO) -> str:

    return file_bytes.read().decode("utf-8", errors="ignore")





def extract_text(uploaded_file) -> str:

    name = uploaded_file.name.lower()

    data = BytesIO(uploaded_file.read())

    if name.endswith(".pdf"):

        return extract_text_from_pdf(data)

    elif name.endswith(".docx"):

        return extract_text_from_docx(data)

    elif name.endswith(".txt"):

        return extract_text_from_txt(data)

    elif name.endswith(".md"):

        return data.read().decode("utf-8", errors="ignore")

    else:

        st.error("Unsupported format. Please upload PDF, DOCX, TXT, or MD.")

        return ""





def markdown_to_pdf_bytes(md_text: str) -> bytes:

    pdf = FPDF()

    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()

    pdf.set_font("Arial", size=11)

    for line in md_text.splitlines():

        pdf.multi_cell(0, 5, line)

    pdf_bytes = BytesIO()

    pdf.output(pdf_bytes)

    pdf_bytes.seek(0)

    return pdf_bytes.getvalue()





# =========================

#  Prompts

# =========================



def build_deep_summary_prompt(doc_text: str, lang: str) -> str:

    if lang == "en":

        language_instruction = "Write the entire output in English."

    else:

        language_instruction = "請使用繁體中文撰寫整份輸出，並以 FDA 審查與合規視角進行深入分析。"



    base = f"""

你是一位具備 FDA 規範、醫藥/醫材審查與戰略規劃專長的「高階策略審閱官」與「知識架構師」。

{language_instruction}



你將收到一份文件內容（可能為藥品、醫療器材、生醫統計、臨床試驗、品質系統、風險管理或其他與 FDA 相關之內容）。

請執行以下任務：



1. 產出一份 **深度、結構化的 Markdown 報告**，長度約 **2,000–3,000 字**。

2. 報告需同時關注：內容本身的邏輯、FDA 合規要點、潛在風險與缺口。

3. 使用以下結構（Markdown 標題）：



# Executive Overview / 文件總覽

- 文件目的、目標對象與核心主題。



## Key Themes & Regulatory Objectives / 關鍵主題與法規目標

- 條列文件欲達成之 FDA 合規或上市策略目標。



## Section-by-Section Analysis / 逐段深度解析

- 針對重要段落或章節，說明：

  - 主要內容在談什麼？

  - 與 FDA 法規、審查觀點的關聯？

  - 潛在風險、疑點或需要補強之處？



## Critical Risks, Gaps, Red Flags / 關鍵風險與缺口

- 從法規、臨床、CMC、統計、安全性、標示與說明書等多面向，點出：

  - 高／中／低風險項目

  - 可能遭 FDA 質疑或要求補件之處。



## Actionable Recommendations / 可執行建議

- 條列具體、可操作之下一步：

  - 例如需補充哪些試驗、補強哪些模組、增加哪些風險控管說明等。



## Stakeholder-Specific Views / 利害關係人視角

- 說明對以下角色的重要解讀：

  - 法規事務（RA）

  - 臨床與醫學團隊

  - 統計與數據科學

  - 品質與藥廠／工廠營運

  - 管理階層／決策者



## Glossary of Key Terms (if applicable) / 專有名詞整理

- 將關鍵 FDA／技術術語條列並做簡明定義。



限制條件：

- 使用 Markdown 標題（#、##、###）與條列。

- 優先避免虛構特定數據；如文件未提供，請以「文件未明確說明」標示。

- 若文件內容不足以支撐結論，需在文中清楚註明不確定性。



以下為文件內容（可能已為長度考量而截斷）：



[DOCUMENT START]

{doc_text[:100000]}

[DOCUMENT END]

"""

    return base.strip()





# =========================

#  Tabs

# =========================



def tab_file_transform_deep_summary():

    st.markdown(f"## {t('tab_file_transform')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)



    uploaded = st.file_uploader(

        t("upload_label"),

        type=["pdf", "docx", "txt"],

        key="file_transform_uploader",

    )



    output_format = st.radio(

        t("output_format"),

        [t("format_markdown"), t("format_pdf")],

        horizontal=True,

        key="output_format_choice",

    )



    if uploaded is not None:

        if st.button(t("run_summary"), type="primary"):

            with st.spinner("Extracting text and generating deep summary…"):

                raw_text = extract_text(uploaded)

                if not raw_text.strip():

                    st.error("No readable text extracted from the file.")

                    st.markdown("</div>", unsafe_allow_html=True)

                    return



                provider, model_id, max_tokens, temperature, system_prompt = get_llm_config()

                lang = st.session_state.get("ui_lang", "zh")



                prompt = build_deep_summary_prompt(raw_text, lang)

                output = call_llm(

                    provider=provider,

                    model=model_id,

                    system_prompt=system_prompt,

                    user_messages=[{"role": "user", "content": prompt}],

                    max_tokens=max_tokens,

                    temperature=temperature,

                )

                if not output:

                    st.markdown("</div>", unsafe_allow_html=True)

                    return



                st.session_state["latest_file_text"] = raw_text

                st.session_state["latest_file_summary_md"] = output

                st.session_state["latest_file_name"] = uploaded.name



                st.markdown("### 📄 Deep Summary (Markdown)")

                st.markdown(output)



                if output_format == t("format_markdown"):

                    st.download_button(

                        "Download Markdown",

                        data=output.encode("utf-8"),

                        file_name=f"{uploaded.name}.summary.md",

                        mime="text/markdown",

                    )

                else:

                    pdf_bytes = markdown_to_pdf_bytes(output)

                    st.download_button(

                        "Download PDF",

                        data=pdf_bytes,

                        file_name=f"{uploaded.name}.summary.pdf",

                        mime="application/pdf",

                    )



    if "latest_file_text" in st.session_state:

        st.markdown("---")

        st.markdown(f"### 💬 {t('chat_with_file')} — {st.session_state.get('latest_file_name', '')}")

        user_q = st.text_area(t("user_prompt"), key="file_chat_prompt")

        if st.button("Ask the file"):

            provider, model_id, max_tokens, temperature, system_prompt = get_llm_config()

            full_context = f"""

以下是原始文件內容與該文件之長篇摘要。請嚴格根據此等資訊作答，若內容不足以支持答案，請明確說明「文件未提供足夠資訊」。



[ORIGINAL DOCUMENT]

{st.session_state['latest_file_text'][:60000]}



[SUMMARY]

{st.session_state['latest_file_summary_md'][:40000]}

"""

            question = user_q.strip()

            if not question:

                st.warning("請輸入問題。")

            else:

                with st.spinner("Thinking with the document…"):

                    answer = call_llm(

                        provider=provider,

                        model=model_id,

                        system_prompt=system_prompt,

                        user_messages=[

                            {"role": "user", "content": full_context},

                            {"role": "user", "content": question},

                        ],

                        max_tokens=max_tokens,

                        temperature=temperature,

                    )

                st.markdown("#### Answer")

                st.markdown(answer or "_No answer produced._")



    st.markdown("</div>", unsafe_allow_html=True)





def tab_file_intelligence():

    st.markdown(f"## {t('tab_file_intel')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    up = st.file_uploader(

        t("upload_label"),

        type=["pdf", "docx", "txt", "md"],

        key="file_intel_uploader",

    )

    if up is not None and st.button("Analyze File"):

        with st.spinner("Analyzing file…"):

            text = extract_text(up)

            provider, model_id, max_tokens, temperature, system_prompt = get_llm_config()

            lang = st.session_state.get("ui_lang", "zh")



            language_instruction = (

                "Write the output in English."

                if lang == "en"

                else "請使用繁體中文撰寫，並以 FDA 審查與合規觀點進行說明。"

            )

            prompt = f"""

你是一位 FDA 法規、臨床與 CMC 整合分析專家。

{language_instruction}



請針對以下文件進行結構化分析，涵蓋：

- 文件目的與適用領域

- 與 FDA 相關的法規或指引（如 21 CFR、GxP、ICH 指南）之關聯

- 潛在風險與缺口

- 建議補強與下一步行動



[DOCUMENT START]

{text[:100000]}

[DOCUMENT END]

"""

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

            st.markdown("### Analysis")

            st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)





def tab_multi_file_synthesis():

    st.markdown(f"## {t('tab_multi_file')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    files = st.file_uploader(

        "Upload multiple files (PDF/DOCX/TXT/MD)",

        type=["pdf", "docx", "txt", "md"],

        accept_multiple_files=True,

        key="multi_files",

    )

    if files and st.button("Combine & Analyze"):

        with st.spinner("Combining and analyzing files…"):

            assembled = []

            for f in files:

                content = extract_text(f)

                assembled.append(

                    f"--- START FILE: {f.name} ---\n{content}\n--- END FILE: {f.name} ---\n"

                )

            combined = "\n".join(assembled)[:150000]



            provider, model_id, max_tokens, temperature, system_prompt = get_llm_config()

            lang = st.session_state.get("ui_lang", "zh")

            language_instruction = (

                "Write the output in English."

                if lang == "en"

                else "請使用繁體中文撰寫，並強調跨文件之 FDA 法規觀點與差異。"

            )



            prompt = f"""

你是一位專精於 FDA 報規與跨文件策略評估的顧問。



{language_instruction}



你將收到多份文件，已以 START/END FILE 標記區分。

請視其為一組「知識庫」，執行以下任務：



- 比較與對照各文件在法規立場、臨床證據、CMC、風險管理等面向的差異與一致性。

- 找出關鍵落差（例如 CTD 模組間前後不一致、統計假設與實際分析不符、說明書與標示不一致等）。

- 產出 Markdown 報告，包含：

  - Executive Summary（整體結論）

  - Cross-Document Comparisons（跨文件比較）

  - Key Risks / Gaps（風險與缺口）

  - FDA 審查觀點下的優先順序與建議下一步



[DOCUMENTS]

{combined}

"""

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

            st.markdown("### Synthesis Report")

            st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)





def tab_smart_replace():

    st.markdown(f"## {t('tab_smart_replace')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        template_text = st.text_area(

            "Template (with placeholders like [Product Name], [Indication])",

            height=260,

        )

    with col2:

        context_text = st.text_area(

            "Context / Raw Data Source (e.g., protocol, CSR, CMC summary)",

            height=260,

        )



    instructions = st.text_area(

        "Natural language instructions (tone, style, constraints)",

        value="請依照 FDA 法規與科學合理性填寫所有欄位，維持專業、精確且審查友善的語氣。",

    )



    if st.button("Run Smart Replace"):

        provider, model_id, max_tokens, temperature, system_prompt = get_llm_config()

        lang = st.session_state.get("ui_lang", "zh")

        language_instruction = (

            "Write the output in English."

            if lang == "en"

            else "請使用繁體中文撰寫完整範本內容。"

        )



        prompt = f"""

你是一位 FDA 報規與法律文本撰寫專家。



{language_instruction}



下列為一份含有占位符的範本（如 [Product Name]、[Indication]、[Dosage] 等）：



[TEMPLATE]

{template_text}



以下為未結構化的背景資料（可能來自 ICH CTD 模組、臨床試驗計畫、CMC 文件、風險管理計畫等）：

[CONTEXT]

{context_text}



使用者給你的額外說明與偏好（語氣、風格、限制）如下：

{instructions}



請依據 CONTEXT 中可合理推論之資訊：

- 補齊所有占位符

- 避免憑空捏造關鍵數據；若文件未提供，請以「（文件未提供明確資訊）」標示

- 調整周邊文字，使全文在語法與法規語氣上自然、連貫

- 以 Markdown 輸出完整且已填寫完成之範本

"""

        with st.spinner("Generating filled template…"):

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

        st.markdown("### Completed Template")

        st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)





def tab_ai_note_keeper():

    st.markdown(f"## {t('tab_note_keeper')}")

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)



    raw_note = st.text_area("Your raw notes / brain dump", height=240, key="note_raw")

    col1, col2, col3, col4, col5 = st.columns(5)

    action = None

    if col1.button("Format"):

        action = "format"

    if col2.button("Tasks"):

        action = "tasks"

    if col3.button("Fix"):

        action = "fix"

    if col4.button("Summary"):

        action = "summary"

    if col5.button("Expand"):

        action = "expand"



    if action and raw_note.strip():

        provider, model_id, max_tokens, temperature, system_prompt = get_llm_config()

        lang = st.session_state.get("ui_lang", "zh")

        language_instruction = (

            "Write the output in English."

            if lang == "en"

            else "請使用繁體中文撰寫，並維持 FDA 報規或專業審查文件常見之語氣。"

        )



        prompt_map = {

            "format": "將這些筆記整理成結構清楚的 Markdown（含標題與條列），方便日後用於 FDA 文件草擬。",

            "tasks": "從這些內容中萃取所有可執行任務，並以核取清單 (- [ ]) 條列，著重於 FDA 報規與合規行動。",

            "fix": "修正文法、用詞與邏輯，使其更適合作為對 FDA 或內部審查使用的專業文字。",

            "summary": "先給出一段精簡 TL;DR 摘要，再以條列方式整理重點與風險項目。",

            "expand": "將簡短的要點擴寫成較完整的段落，並加入 FDA 合規觀點或實務建議。",

        }

        prompt = f"""

你是一位專門協助 FDA 報規團隊整理思路的「知識管理顧問」。



{language_instruction}



使用者的原始筆記如下：

{raw_note}



任務：{prompt_map[action]}



請只輸出整理後的 Markdown 筆記。

"""

        with st.spinner("Transforming notes…"):

            result = call_llm(

                provider=provider,

                model=model_id,

                system_prompt=system_prompt,

                user_messages=[{"role": "user", "content": prompt}],

                max_tokens=max_tokens,

                temperature=temperature,

            )

        st.markdown("### Transformed Notes")

        st.markdown(result or "_No output._")

    st.markdown("</div>", unsafe_allow_html=True)





# =========================

#  Main

# =========================



def main():

    st.set_page_config(

        page_title="AuditFlow AI · Masterpiece Edition (FDA)",

        layout="wide",

    )



    # Init session defaults

    if "ui_lang" not in st.session_state:

        st.session_state.ui_lang = "zh"

    if "dark_mode" not in st.session_state:

        st.session_state.dark_mode = True

    if "artist_style_key" not in st.session_state:

        st.session_state.artist_style_key = "van_gogh"



    # Load agents

    agents = load_agents()



    # Sidebar global controls

    with st.sidebar:

        st.markdown("## 🌐 Global Settings")

        lang_label = st.radio("Language / 語言", ["English", "繁體中文"], key="lang_radio")

        st.session_state.ui_lang = "en" if lang_label == "English" else "zh"



        dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode_toggle")

        st.session_state.dark_mode = dark_mode



        active_style = style_selector_ui()

        render_api_key_inputs()

        selected_agent = agent_selector_ui(agents)

        render_llm_controls()



    # Apply painter theme

    apply_theme(active_style, st.session_state.dark_mode)



    # Header

    st.markdown(f"<h1 class='accent-title'>{t('app_title')}</h1>", unsafe_allow_html=True)

    st.markdown(t("subtitle"))

    if selected_agent:

        st.markdown(

            f"<div class='accent-chip'>目前代理人：{selected_agent.get('display_name_zh','')}</div>",

            unsafe_allow_html=True,

        )



    tab1, tab2, tab3, tab4, tab5 = st.tabs([

        t("tab_file_transform"),

        t("tab_file_intel"),

        t("tab_multi_file"),

        t("tab_smart_replace"),

        t("tab_note_keeper"),

    ])



    with tab1:

        tab_file_transform_deep_summary()

    with tab2:

        tab_file_intelligence()

    with tab3:

        tab_multi_file_synthesis()

    with tab4:

        tab_smart_replace()

    with tab5:

        tab_ai_note_keeper()





if __name__ == "__main__":

    main()
