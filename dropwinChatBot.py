import sys
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
# from langchain_community.llms import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document

st.write(f"Python version: {sys.version}")

# 문서 예시 - 스포츠 경매 데이터 예시 (여기선 임시로 설정)
sports_auction_info = [
    {
        "경매번호": "2025-07-16-001",
        "종목": "축구",
        "선수명": "홍길동",
        "소속팀": "서울FC",
        "경매시작가": "1억원",
        "마감일": "2025-07-20",
        "특이사항": "K리그 득점왕 수상 경력 있음"
    },
    {
        "경매번호": "2025-07-16-002",
        "종목": "야구",
        "선수명": "김철수",
        "소속팀": "부산베어스",
        "경매시작가": "1억 5천만원",
        "마감일": "2025-07-22",
        "특이사항": "골든글러브 3회 수상자"
    }
]

# 문서화
documents = [
    Document(
        page_content=", ".join([
            f"{key}: {value}" for key, value in item.items()
        ])
    )
    for item in sports_auction_info
]

# 임베딩 & 벡터DB 생성
# 임베딩 & 벡터DB 생성
embedding_function = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)
db = FAISS.from_documents(documents, embedding_function)

retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

# Ollama 모델 설정
llm = ChatOllama(model="gemma3", temperature=0.5, base_url="http://127.0.0.1:11434/")

template = """
너는 DropWin 웹 경매 사이트의 스포츠 경매 전문 상담 직원이야.
사용자와 대화할 때는 반드시 정중하고 친절한 말투를 사용해.
답변은 항상 한국어로 해줘.

📌 다음과 같은 기준을 지켜서 답변해 줘:
1. 질문에 최대한 정확하고 구체적으로 답변해.
2. 숫자, 날짜, 경매 정보가 있으면 정확히 전달해.
3. 만약 데이터에 해당 정보가 없으면 "해당 정보는 확인되지 않습니다"라고 정중하게 안내해.
4. 답변 마지막에는 간단한 안내 문구(예: "더 궁금한 게 있으시면 언제든지 물어보세요!")를 추가해.

📝 아래는 질문과 관련된 참고 정보야. 이 정보만 바탕으로 답변해.
만약 참고 정보에 없는 내용이더라도 거짓으로 대답하지 말고, 솔직하게 없다고 말해줘.

<사용자 질문에 대한 참고 정보>
{context}

질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 체인 구성
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm

# --- Streamlit UI 시작 ---
st.title("🏅 스포츠 경매 챗봇")
st.write("스포츠 스타 경매에 대해 궁금한 점을 물어보세요!")

# 채팅 히스토리 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 새 채팅 버튼
if st.button("🔄 새 채팅 시작"):
    st.session_state.chat_history = []
    st.rerun()

# 사용자 질문 입력
user_input = st.text_input("질문을 입력하세요:")

# 사용자 질문 처리
if user_input:
    response = chain.invoke({'question': user_input}).content
    st.session_state.chat_history.append(("🙂 사용자", user_input))
    st.session_state.chat_history.append(("🤖 챗봇", response))

# 이전 채팅 내역 출력
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 이전 대화")
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")