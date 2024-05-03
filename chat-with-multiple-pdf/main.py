import streamlit as st

from app import ChatWithMultiplePDF

CWMP=ChatWithMultiplePDF()

def main():
    st.set_page_config(page_title="Chat with Multiple PDF", page_icon=":robot:")

    st.header("Chat with Multiple PDF using Gemini 💁")

    user_question = st.text_input("Ask a question about the PDFs")

    if user_question:
        resp=CWMP.user_input(user_question)
        st.write("Reply: ", resp)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text=CWMP.get_pdf_text(pdf_docs)
                text_chunks=CWMP.get_text_chunks(raw_text)
                CWMP.get_vector_store(text_chunks)
                st.success("Processing complete!")

if __name__=="__main__":
    main()