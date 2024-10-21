from langchain.schema import Document
from langgraph.graph import END
from langgraph.graph import StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
import os
from credentials import TAVLY_KEY
import json
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from typing import TypedDict
from typing import List, Annotated
from prompts import *
from langchain_community.tools.tavily_search import TavilySearchResults

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class SimpleRag:

    def __init__(self):
        local_llm = "llama3.2:3b-instruct-fp16"
        self.llm = ChatOllama(model=local_llm, temperature=0)
        self.llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
        os.environ["TAVILY_API_KEY"] = TAVLY_KEY
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        )

        # Create retriever
        self.retriever = vectorstore.as_retriever(k=3)

        # Web Search from Tavily
        self.web_search_tool = TavilySearchResults(k=3)


    ### Nodes
    def retrieve(self, state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Write retrieved documents to documents key in state
        documents = self.retriever.invoke(question)
        return {"documents": documents}


    def generate(self, state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        docs_txt = format_docs(documents)
        rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}


    def grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            doc_grader_prompt_formatted = doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "web_search": web_search}


    def web_search(self,state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        return {"documents": documents}


    ### Edges
    def route_question(self,state):
        """
        Route question to web search or RAG

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        route_question = self.llm_json_mode.invoke(
            [SystemMessage(content=router_instructions)]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state["web_search"]
        filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=format_docs(documents), generation=generation.content
        )
        result = self.llm_json_mode.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            # Test using question and generation from above
            answer_grader_prompt_formatted = answer_grader_prompt.format(
                question=question, generation=generation.content
            )
            result = self.llm_json_mode.invoke(
                [SystemMessage(content=answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    def run(self): 

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("websearch", self.web_search)  # web search
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate

        # Build graph
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
                "max retries": END,
            },
        )

        # Compile
        print("compiling...")
        graph = workflow.compile()

        print("running through rag...")
        inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
        for event in graph.stream(inputs, stream_mode="values"):
            print(event)

if __name__=='__main__':
    SimpleRag().run()