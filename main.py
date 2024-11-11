import streamlit as st
import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import arxiv
from neo4j import GraphDatabase
from transformers import pipeline
import PyPDF2
from neo4j import GraphDatabase, RoutingControl

# Data Models
class Paper(BaseModel):
    id: str
    title: str
    authors: List[str]
    summary: str
    published_date: datetime
    url: str
    category: str

class SearchQuery(BaseModel):
    topic: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None

# Agent Classes
class SearchAgent:
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(self, query: str, max_results: int = 2) -> List[Paper]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        cwd = os.getcwd()
        os.makedirs(os.path.join(cwd, 'papers'), exist_ok=True)
        download_dir_path = os.path.join(cwd , 'papers')
        db = Database("neo4j://localhost:7687", "neo4j", "password")
        papers = []
        id = 0
        for result in self.client.results(search):
            paper_pdf = next(arxiv.Search(id_list=[result.entry_id[21:]]).results())

            paper = Paper(
                id=result.entry_id[21:],
                title=result.title,
                authors=[str(author) for author in result.authors],
                summary=result.summary,
                published_date=result.published,
                url=result.pdf_url,
                category=result.primary_category
            )

            try:
                # temporarily storing pdf and extracting relevant information
                paper_pdf.download_pdf(dirpath=download_dir_path , filename=f"{paper.id}.pdf")
                text = db.extract_data(os.path.join(download_dir_path , f"{paper.id}.pdf" ))
                db.add_paper(paper=paper , text = text)
                os.remove(os.path.join(download_dir_path , f"{paper.id}.pdf" ))
                papers.append(paper)
                id+=1
            except FileNotFoundError:
                pass
        os.rmdir(download_dir_path)
        return papers

class Database:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def extract_data(self , id):

        text = ""
        pdf_path = os.path.join(os.getcwd(), 'papers', f'{id}.pdf')
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

        return text

    def close(self):
        self.driver.close()

    def add_paper(self, paper , text):
        with self.driver.session() as session:
            session.execute_write(self._add_paper_tx, paper , text)

    @staticmethod
    def _add_paper_tx(tx, paper , text):
        query = (
            "MERGE (p:Paper {id: $id}) "
            "ON CREATE SET p.url = $url, p.summary = $summary, p.text = $text "
            "WITH p "
            "UNWIND $authors AS author_name "
            "MERGE (a:Author {name: author_name}) "
            "MERGE (p)-[:WRITTEN_BY]->(a)"
        )
        tx.run(query, id=paper.id, url=paper.url, summary=paper.summary, authors=paper.authors , text = text)

    def query_paper(self, id):
        with self.driver.session() as session:
            result = session.execute_read(self.query_paper_tx, id)
            if result:
                return result
            else:
                return ""

    @staticmethod
    def query_paper_tx(tx, id) -> str:
        query = (
            "MATCH (p:Paper {id: $id}) "
            "RETURN p.text AS text"
        )
        result = tx.run(query, id=id)

def search_papers(query: SearchQuery):
    search_agent = SearchAgent()
    papers = search_agent.search_papers(query.topic)
    return papers


class Query_Agent:
    def __init__(self):
        # Initialize a pipeline for question-answering , summarization and review
        # model_qa = "huawei-noah/TinyBERT_General_4L_312D"
        # model_qa = 'google/electra-base-discriminator'
        # model_review = "allenai/led-base-16384"
        model_qa = "bert-large-uncased-whole-word-masking-finetuned-squad"
        model_summary = "sshleifer/distilbart-cnn-12-6"
        model_review = "google/bigbird-pegasus-large-arxiv"
        self.qa_pipeline = pipeline("question-answering" , model = model_qa)
        self.summarization_pipeline = pipeline("summarization" , model = model_summary)
        self.review_pipeline = pipeline("summarization" , model = model_review)

    def answer_question(self , question, context):
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]

    def summarize_text(self , text):
        summary = self.summarization_pipeline(text[:1000], max_length=100, min_length=25, do_sample=False)
        return ''.join(summary[0]["summary_text"].split('.')[:-1])

    def review_text(self , text):
        max_len = min(len(text) , 15000)
        review = self.review_pipeline(text[:max_len], max_length=250, min_length=50, do_sample=False)
        return ''.join(review[0]["summary_text"].split('.')[:-1])

def get_llm_response(query: str, papers: List[Paper] , type: str) -> str:
    # Construct context from selected papers
    db = Database("neo4j://localhost:7687", "neo4j", "password")
    papers_context = "\n\n".join([
        f"Title: {paper.title}\nAuthors: {paper.authors}\nAbstract: {paper.summary} Text: {db.extract_data(paper.id)}"
        for paper in papers
    ])
    
    AI_agent = Query_Agent()
    try:
        if type == "Summary" : 
            response = AI_agent.summarize_text(papers_context)
        elif type == "QNA":
            response = AI_agent.answer_question(query , papers_context)
        else:
            response = AI_agent.review_text(papers_context)
        return response
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"

def papers_search_page():
    st.title("Academic Research Paper Assistant")
    
    # Initialize session state for selected papers if it doesn't exist
    if 'selected_papers' not in st.session_state:
        st.session_state.selected_papers = []
    
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'search'
    
    # Sidebar for search parameters
    with st.sidebar:
        st.header("Search Parameters")
        topic = st.text_input("Research Topic")
        start_year = st.number_input("Start Year", min_value=2000, max_value=2024, value=2020)
        end_year = st.number_input("End Year", min_value=2000, max_value=2024, value=2024)
        search_button = st.button("Search Papers")
    
    if search_button:
        query = SearchQuery(topic=topic, start_year=start_year, end_year=end_year)
        papers = search_papers(query=query)
        st.session_state.papers = papers
        st.session_state.selected_papers = [False] * len(papers)

    if st.session_state.papers:
        st.header("Research Papers")
        
        for i, paper in enumerate(st.session_state.papers):
            with st.container():
                st.markdown(f"### {paper.title}")
                st.markdown(f"_**Authors**: {paper.authors}_")
                st.write(f"**Abstract:** {paper.summary}")
                st.markdown(f"[Read more]({paper.url})", unsafe_allow_html=True)
                
                checkbox_key = f"paper_{i}"
                selected = st.checkbox(
                    "Select for discussion",
                    key=checkbox_key,
                    value=st.session_state.selected_papers[i]
                )
                
                st.session_state.selected_papers[i] = selected
                st.markdown("---")
        
        # Only show the proceed button if at least one paper is selected
        if any(st.session_state.selected_papers):
            if st.button("Proceed with Selected Papers"):
                st.session_state.current_page = 'query'
                st.rerun()

def query_papers_page():
    st.title("Query Selected Papers")
    
    # Get selected papers
    selected_papers = [
        paper for paper, selected in zip(st.session_state.papers, st.session_state.selected_papers)
        if selected
    ]
    
    # Display selected papers
    st.header("Selected Papers")
    for paper in selected_papers:
        st.markdown(f"- **{paper.title}** by {paper.authors}")
    
    # Query input
    st.header("Ask Questions")
    query = st.text_area("Enter your question about the selected papers:", height=100)
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.current_page = 'search'
            st.rerun()
    
    # Submit button for query
    if st.button("Question Answer"):
        if query:
            with st.spinner("Getting answer..."):
                response = get_llm_response(query, selected_papers , type = "QNA")
                st.markdown("### Answer")
                st.write(response)
        else:
            st.warning("Please enter a question.")
    
    if st.button("Summarize"):
        with st.spinner("Getting answer..."):
            response = get_llm_response(query, selected_papers , type = "Summary")
            st.markdown("### Answer")
            st.write(response)
    
    if st.button("Review"):
        with st.spinner("Getting answer..."):
            response = get_llm_response(query, selected_papers , type = "Review")
            st.markdown("### Answer")
            st.write(response)

def main():

    # Page routing
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'search'
    
    if st.session_state.current_page == 'search':
        papers_search_page()
    elif st.session_state.current_page == 'query':
        query_papers_page()

if __name__ == "__main__":
    main()