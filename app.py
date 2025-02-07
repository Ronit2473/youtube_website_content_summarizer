import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter URL (YouTube or Website)", label_visibility="collapsed")

# Initialize the Groq model
if groq_api_key.strip():
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq API: {e}")
        llm = None
else:
    llm = None

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to trigger summarization
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the Groq API Key and a URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can either be a YouTube video URL or a website URL.")
    else:
        if llm:
            try:
                with st.spinner("Loading content..."):
                    # Load content based on URL type
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                        )
                    
                    docs = loader.load()

                    # Chain for Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success("Summary:")
                    st.write(output_summary)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Groq API Key is not valid or failed to initialize the model.")
