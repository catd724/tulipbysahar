import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter////
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
# Configure the API key using Streamlit secrets
googlekey = st.secrets["auth_token"]
genai.configure(api_key=googlekey)

# Initialize chat history and current QA state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""

if "current_answer" not in st.session_state:
    st.session_state["current_answer"] = ""

if "current_image" not in st.session_state:
    st.session_state["current_image"] = None

# Dictionary mapping keywords to image paths
keyword_image_dict = {
    "An Unforgettable Love": "https://tulipbysahar.com/wp-content/uploads/2023/10/1-1-300x300.jpg",
    "Autumn Tales": "https://tulipbysahar.com/wp-content/uploads/2023/10/3-1-300x300.jpg",
    "Chamomile A Healer": "https://tulipbysahar.com/wp-content/uploads/2024/03/4-300x300.jpg",
    "Euphoria": "https://tulipbysahar.com/wp-content/uploads/2023/02/2-300x300.jpg",
    "Fairy Dust Florals": "https://tulipbysahar.com/wp-content/uploads/2023/10/5-1-300x300.jpg",
    "Fairy Realm": "https://tulipbysahar.com/wp-content/uploads/2024/03/10-300x300.jpg",
    "Feminine": "https://tulipbysahar.com/wp-content/uploads/2023/02/28-300x300.jpg",
    "Fern": "https://tulipbysahar.com/wp-content/uploads/2024/03/25-300x300.jpg",
    "Forget me not": "https://tulipbysahar.com/wp-content/uploads/2023/02/12-300x300.jpg",
    "Lavender : A Forgiving Heart": "https://tulipbysahar.com/wp-content/uploads/2023/10/2-3-300x300.jpg",
    "Lost Forest": "https://tulipbysahar.com/wp-content/uploads/2023/10/Untitled-design-2-300x300.jpg",
    "Lunula": "https://tulipbysahar.com/wp-content/uploads/2023/10/Untitled-design-1-300x300.jpg",
    "Lunula Pair": "https://tulipbysahar.com/wp-content/uploads/2023/10/10-1-300x300.jpg",
    "Maiden of Snow": "https://tulipbysahar.com/wp-content/uploads/2024/03/1-300x300.jpg",
    "Marigold Guide to living": "https://tulipbysahar.com/wp-content/uploads/2024/03/14-300x300.jpg",
    "My Hearts Song": "https://tulipbysahar.com/wp-content/uploads/2023/10/7-300x300.jpg",
    "My Rose Tinted Dream": "https://tulipbysahar.com/wp-content/uploads/2024/03/22-300x300.jpg",
    "Oceans Heart": "https://tulipbysahar.com/wp-content/uploads/2023/02/10-300x300.jpg",
    "Promised Love": "https://tulipbysahar.com/wp-content/uploads/2023/10/1-300x300.jpg",
    "Rose Bud": "https://tulipbysahar.com/wp-content/uploads/2024/03/30-300x300.jpg",
    "Rose Romantica": "https://tulipbysahar.com/wp-content/uploads/2023/02/5-300x300.jpg",
    "Serene": "https://tulipbysahar.com/wp-content/uploads/2023/02/3-300x300.jpg",
    "Sisterhood Set": "https://tulipbysahar.com/wp-content/uploads/2023/02/23-300x300.jpg",
    "Sunflower": "https://tulipbysahar.com/wp-content/uploads/2024/03/27-300x300.jpg",
    "Unspoken Love": "https://tulipbysahar.com/wp-content/uploads/2023/10/Untitled-design-300x300.jpg",
    "Witchs Whisper": "https://tulipbysahar.com/wp-content/uploads/2023/02/14-300x300.jpg",
    "z.Botanica": "https://tulipbysahar.com/wp-content/uploads/2023/02/11-300x300.jpg",
    "z.Daisy": "https://tulipbysahar.com/wp-content/uploads/2023/02/6-300x300.jpg",
    "z.Daisy Innocence": "https://tulipbysahar.com/wp-content/uploads/2023/02/30-300x300.jpg",
    "z.Queen Annes lace": "https://tulipbysahar.com/wp-content/uploads/2023/02/27-300x300.jpg",
    "Golden Hour": "https://tulipbysahar.com/wp-content/uploads/2023/03/rev.jpg",
    "JellyBellyBop": "https://tulipbysahar.com/wp-content/uploads/2023/02/3-1-300x300.jpg",
    "Past Life Blooms": "https://tulipbysahar.com/wp-content/uploads/2023/02/key1.jpg",
    "Sun Shine": "https://tulipbysahar.com/wp-content/uploads/2023/02/key3.jpg"
    # Add more keyword-image mappings as needed
}


# def get_pdf_text(pdf_docs):
#     """Extracts text from a list of PDF documents."""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     """Splits text into chunks for processing."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     """Creates a vector store from text chunks."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Creates a conversational chain for question answering."""
    prompt_template = """
    You are 'Tulip', a customer service provider at 'tulip by sahar'. Answer the question as detailed as possible from the provided context, make sure to provide all the details, don't provide the wrong answer, if the user asks any question that has a relatable answer in the provided context provide the answer and also greet the user when he greets, not on every answer. If the customer asks for a specific product then also provide him with the link of that product and also the image link. If the customer asks for image of a product just write the name of the product without any special characters and commas but include spaces in response  \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Processes user input and updates the chat history and current answer."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Update current question and answer
    st.session_state["current_question"] = user_question
    st.session_state["current_answer"] = response["output_text"]
    
    # Check for keywords in the response and update the current image
    for keyword, image_path in keyword_image_dict.items():
        if keyword.lower() in response["output_text"].lower():
            st.session_state["current_image"] = image_path
            st.session_state["image_cap"] = keyword

            break
    else:
        st.session_state["current_image"] = None
    
    # Update chat history
    st.session_state["chat_history"].append({"question": user_question, "answer": response["output_text"]})

def display_chat_history():
    """Displays the chat history."""
    if st.session_state["chat_history"]:
        st.subheader("Previous Questions:")
        for chat in st.session_state["chat_history"][:-1]:
            st.write(f"**You:** {chat['question']}")
            st.write(f"**Tulip:** {chat['answer']}")
    else:
        st.write("No conversations yet.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="ChatBot")
    st.header("ChatBot @ TulipBySahar")

    # Use a form for user input
    with st.form(key='user_input_form'):
        user_question = st.text_input("Ask a Question")
        submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_question:
            user_input(user_question)
            
    # Display current question and answer
    if st.session_state["current_question"]:
        st.write(f"**Tulip:** {st.session_state['current_answer']}")
        st.write("")  # Adding an empty line after the current answer
        
        # Display the image if available
        if st.session_state["current_image"]:
            st.image(st.session_state["current_image"], caption=st.session_state["image_cap"],width=200)

    display_chat_history()

    # with st.sidebar:
    #     st.title("Menu:")
    #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    #     if st.button("Submit & Process"):
    #         if pdf_docs:
    #             with st.spinner("Processing..."):
    #                 raw_text = get_pdf_text(pdf_docs)
    #                 text_chunks = get_text_chunks(raw_text)
    #                 get_vector_store(text_chunks)
    #                 st.success("Processing complete. You can now ask questions about the content.")
    #         else:
    #             st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
