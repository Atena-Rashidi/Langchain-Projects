import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


## Function to get response from LLM: Llama 2 model
def get_llama_response(input_text, num_of_words, blog_style):
    ### LLama2 Model
    llm = CTransformers(model="llama-2-7b-chat-q4_0.gguf",
                        model_type="llama",
                        temprature=0.01,
                        max_new_tokens=1024,
                        top_p=0.9,
                        repetition_penalty=1.03)
    
    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {num_of_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_of_words"],
                            template=template)
    

    ## Generate the response from the model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, num_of_words=num_of_words))

    return response



st.set_page_config(page_title="Generate Blogs",
                   page_icon=":robot:",
                   layout="centered",
                   initial_sidebar_state="collapsed")


st.header("Generate Blogs :robot:")


input_text = st.text_input("Enter the Blog Topic")
# Creating a 2 button for additional features
colum1, colum2 = st.columns([5, 5])
with colum1:
    num_of_words = st.text_input("Number of Words")
with colum2:
    blog_style = st.selectbox("Writing the blog for",
                              ("Researchers", "Data Scientist", "Common People"), index = 0)
    
    # Creating a button
submit = st.button("Generate")


## Final response
if submit:
    st.write(get_llama_response(input_text,num_of_words,blog_style))