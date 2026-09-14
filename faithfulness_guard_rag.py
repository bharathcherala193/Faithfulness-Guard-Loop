from langchain_community.document_loaders import PyPDFLoader
import os
import math
import time
import tempfile
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Annotated


class State(TypedDict):
    input: str
    search_query: str
    messages: Annotated[list, add_messages]
    embedding: list
    first_response: str
    grounded: bool
    scores: Dict[str, float]
    attempt: int
    max_attempts: int
    strict: bool
    enable_deep_eval: bool
    enable_grounding_check: bool
    action: str
    history: list[Dict]
    latency: float


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(time.time())

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = InMemorySaver()

st.set_page_config(page_title="Faithfulness-Guard RAG", page_icon="🛡️")
st.title("Faithfulness-Guard RAG Agent")

uploaded_file = st.file_uploader(
    "Upload a PDF document to train the agent",
    type=["pdf"],
    help="Only PDF documents are supported",
    accept_multiple_files=False
)

if "retriever" not in st.session_state:
    if uploaded_file:
        with st.spinner("Processing Documents..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            # Bigger chunks than before - 400 chars was cutting explanations
            # off mid-thought, which is why "explain about it" style
            # questions were coming back thin or contradictory. 900/150
            # keeps a fuller unit of meaning per chunk while still being
            # small enough to answer from quickly.
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            chunks = splitter.split_documents(documents)
            embedding = OllamaEmbeddings(model="nomic-embed-text")
            vector = Chroma.from_documents(chunks, embedding)
            st.session_state.retriever = vector
            os.remove(tmp_path)

if "retriever" not in st.session_state:
    st.warning("Please upload a PDF first")
    st.stop()


def rewrite_query(state: State):
    # Vague follow-ups ("explain about it", "what's the project") embed
    # poorly on their own - their wording doesn't match the document's
    # actual language, so retrieval grabs the wrong chunks. This step uses
    # the recent chat history to turn the question into something
    # specific enough to search well, BEFORE it goes anywhere near the
    # vector store.
    recent = state['messages'][-6:]
    convo_text = "\n".join(f"{m.type}: {m.content}" for m in recent)
    llm = ChatOllama(model="llama3.1:8b", temperature=0, num_predict=60)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the LATEST MESSAGE into a clear, standalone search query "
         "for a document search engine. Use the conversation so far only to "
         "resolve vague words like 'it', 'the project', or 'this' into "
         "something specific. If the latest message is already specific, "
         "just repeat it as-is. Reply with ONLY the rewritten query, "
         "nothing else - no explanation, no quotes.\n\n"
         "Conversation so far:\n{convo}\n\n"
         "Latest message: {question}")
    ])
    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"convo": convo_text, "question": state['input']}).strip()
    return {"search_query": rewritten or state['input']}


def embed(state: State):
    # k is fixed on purpose. This loop never changes HOW MUCH is retrieved -
    # only how strictly the model is told to stick to what it already has.
    retriv = st.session_state.retriever.as_retriever(search_kwargs={"k": 6})
    query = state.get("search_query") or state['input']
    res = retriv.invoke(query)
    final = [doc.page_content for doc in res]
    return {"embedding": final}


def agent_builder(state: State):
    start_time = time.time()
    llm = ChatOllama(model="llama3.1:8b", temperature=0.2, num_predict=400)
    context = "\n\n".join(state['embedding'])

    if state.get("strict", False):
        # Only used on retry, after the quick_check flagged the first answer
        # as ungrounded. Same context, same question - just a harder rule.
        # Important: only fall back to "Not found" when NOTHING in the
        # question is answerable - otherwise just answer the supported
        # part directly, with no disclaimer prefix. The earlier version of
        # this prompt caused the model to say "Not found" and then answer
        # anyway, which reads as contradictory.
        system_text = (
            "You are a careful assistant. Answer using ONLY information "
            "explicitly present in the context below. Answer directly, with "
            "no preamble or disclaimer. If part of the question isn't "
            "covered by the context, just leave that part out - don't "
            "mention that it's missing. Only reply with exactly 'Not found "
            "in the document.' if NONE of the question can be answered "
            "from the context at all.\n\n"
            "Context:\n{context}"
        )
    else:
        system_text = (
            "You are a helpful assistant. Use the following pieces of retrieved "
            "context to answer the user's question clearly and concisely. If you "
            "don't know the answer, just say that you don't know.\n\n"
            "Context:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"messages": state['messages'], "context": context})
    latency = time.time() - start_time
    return {"first_response": response, "latency": latency}


def quick_check(state: State):
    # This replaces the old always-run RAGAS call in the main loop. It's ONE
    # short call to the same model that already answered (no second big
    # model to load, no multi-step RAGAS decomposition) - just a yes/no
    # groundedness check. This is what keeps the whole thing fast while
    # still catching answers that drifted from the retrieved context.
    llm = ChatOllama(model="llama3.1:8b", temperature=0, num_predict=5)
    context = "\n\n".join(state['embedding'])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict fact-checker. Reply with exactly one word: "
         "YES if every claim in the ANSWER is directly supported by the "
         "CONTEXT, or NO if the ANSWER includes anything not found in the "
         "CONTEXT.\n\nContext:\n{context}\n\nAnswer:\n{answer}")
    ])
    chain = prompt | llm | StrOutputParser()
    verdict = chain.invoke({"context": context, "answer": state['first_response']})
    grounded = verdict.strip().upper().startswith("Y")
    return {"grounded": grounded}


def quick_decision(state: State):
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 2)

    history = list(state.get("history", []))
    history.append({
        "attempt": attempt,
        "grounded": state.get("grounded", True),
        "strict": bool(state.get("strict", False)),
        "latency": float(state.get("latency", 0))
    })

    if state.get("grounded", True) or attempt >= max_attempts:
        return {"action": "stop", "history": history}
    return {"attempt": attempt + 1, "strict": True, "action": "retry", "history": history}


def deep_evaluate(state: State):
    # Optional, opt-in only (checkbox). This is the slow, thorough RAGAS
    # pass with a separate 12B judge model - useful when you want real
    # numeric faithfulness/relevancy scores, not on the fast path.
    llm = LangchainLLMWrapper(ChatOllama(model="gemma3:12b", temperature=0, format="json", timeout=600))
    embedding = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    data = {
        'question': [state['input']],
        'answer': [state['first_response']],
        'contexts': [state['embedding']]
    }
    dataset = Dataset.from_dict(data)
    try:
        evaluation = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embedding
        )
        raw = evaluation.scores[0]
        faith = float(raw.get("faithfulness", 0.0))
        relevance = float(raw.get("answer_relevancy", 0.0))
        if math.isnan(faith): faith = None
        if math.isnan(relevance): relevance = None
    except Exception:
        faith = None
        relevance = None
    return {"scores": {"faithfulness": faith, "answer_relevancy": relevance}}


def get_graph():
    builder = StateGraph(State)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("embed", embed)
    builder.add_node("agent_builder", agent_builder)
    builder.add_node("quick_check", quick_check)
    builder.add_node("quick_decision", quick_decision)
    builder.add_node("deep_evaluate", deep_evaluate)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "embed")
    builder.add_edge("embed", "agent_builder")

    def after_answer(state: State):
        # By default this goes straight to "stop" - one clean answer, no
        # retry prompt weirdness. The grounding check only runs at all if
        # the sidebar checkbox turned it on.
        if state.get("enable_grounding_check", False):
            return "check"
        return "deep" if state.get("enable_deep_eval", False) else "stop"

    builder.add_conditional_edges(
        "agent_builder",
        after_answer,
        {
            "check": "quick_check",
            "deep": "deep_evaluate",
            "stop": END
        }
    )
    builder.add_edge("quick_check", "quick_decision")

    def route(state: State):
        if state.get("action") == "retry":
            return "retry"
        return "deep" if state.get("enable_deep_eval", False) else "stop"

    builder.add_conditional_edges(
        "quick_decision",
        route,
        {
            "retry": "agent_builder",   # regenerate only - do NOT go back to "embed"
            "deep": "deep_evaluate",    # optional, only when the checkbox is on
            "stop": END
        }
    )
    builder.add_edge("deep_evaluate", END)
    return builder.compile(checkpointer=st.session_state.agent_memory)


graph = get_graph()

with st.sidebar:
    enable_grounding_check = st.checkbox(
        "Enable grounding retry check",
        value=False,
        help="Off by default - you get one direct answer, no retry. Turn "
             "this on to add a fast yes/no groundedness check that retries "
             "once with a stricter prompt if the answer isn't backed by "
             "the retrieved text."
    )
    enable_deep_eval = st.checkbox(
        "Run deep RAGAS check too (slower)",
        value=False,
        help="Runs full RAGAS scoring with a second, larger model after "
             "the answer - adds real time, use only when you want the "
             "numeric faithfulness/relevancy scores."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

if input := st.chat_input("Ask a question about the uploaded document"):
    with st.chat_message("user"):
        st.markdown(input)
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            response = graph.invoke({
                "input": input,
                "messages": [HumanMessage(content=input)],
                "attempt": 1,
                "max_attempts": 2,
                "strict": False,
                "enable_deep_eval": enable_deep_eval,
                "enable_grounding_check": enable_grounding_check,
                "history": []
            },
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            })
            total_latency = time.time() - start
            final_answer = response.get("first_response")
            history = response.get("history", [])
            if len(history) > 1:
                st.markdown("✔ Regenerated with stricter grounding")
            st.markdown(final_answer)
            st.write("Attempts:", len(history))
            st.markdown(f"Total Latency: {total_latency:.2f} seconds")

            with st.sidebar:
                st.caption("Quick groundedness check:")
                with st.expander("Attempt history"):
                    for h in history:
                        st.write(
                            f"Attempt {h['attempt']} | "
                            f"strict={h['strict']} | "
                            f"grounded={h['grounded']} | "
                            f"latency={h['latency']:.2f}s"
                        )
                with st.expander("Retrieved context (debug)"):
                    st.write("Search query used:", response.get("search_query"))
                    for i, chunk in enumerate(response.get("embedding", [])):
                        st.text_area(f"Chunk {i+1}", chunk, height=100)
                if enable_deep_eval and response.get("scores"):
                    st.caption("Deep RAGAS scores:")
                    st.write(response.get("scores"))

            st.session_state.messages.append({"role": "assistant", "content": final_answer})