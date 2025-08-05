import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# --- LangChain 관련 라이브러리 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda

# --- 0. 페이지 기본 설정 및 스타일 ---
st.set_page_config(page_title="스타트업 네비게이터", page_icon="🧭")

# 사용자 말풍선 스타일을 위한 CSS
st.markdown("""
<style>
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #DCF8C6; /* 연두색 배경 */
        color: black;
        border-radius: 15px;
        padding: 10px 15px;
        display: inline-block;
        max-width: 80%;
        position: relative;
        margin-right: 10px;
        text-align: left;
    }
    .user-message:after {
        content: '';
        position: absolute;
        top: 10px;
        right: -10px;
        width: 0;
        height: 0;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        border-left: 10px solid #DCF8C6;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. 데이터 및 LangChain 컴포넌트 로드 ---
@st.cache_resource
def load_data_and_components():
    """미리 생성된 데이터프레임과 FAISS 벡터 저장소를 로드합니다."""
    try:
        df = pd.read_pickle('vector_store.pkl')
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return df, retriever
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}. build_vector_store.py를 먼저 실행했는지 확인해주세요.")
        return None, None

# --- .env 파일에서 API 키 로드 ---
load_dotenv()
if not os.getenv('OPENAI_API_KEY'):
    st.error("OpenAI API 키를 .env 파일에 설정해주세요.")
    st.stop()

# --- 데이터 및 LangChain 컴포넌트 로드 ---
all_books_df, retriever = load_data_and_components()
if retriever is None:
    st.stop()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 이미지 URL ---
COVER_IMAGES = {
    '린 스타트업': 'https://image.yes24.com/goods/7921251/XL',
    '제로 투 원': 'https://image.yes24.com/goods/103990890/XL',
    '비즈니스 아이디어의 탄생': 'https://image.yes24.com/goods/91868851/XL',
    '기업 창업가 매뉴얼': 'https://image.yes24.com/goods/11928450/XL',
    '아이디어 불패의 법칙': 'https://image.yes24.com/goods/89707566/XL',
    '그냥 하는 사람': 'https://image.yes24.com/goods/146284662/XL',
    '브랜드 창업 마스터': 'https://image.yes24.com/goods/148175776/XL',
    '창업이 막막할 때 필요한 책': 'https://image.yes24.com/goods/147973900/XL',
    '마케팅 설계자': 'https://image.yes24.com/goods/116255710/XL',
    '스타트업 설계자': 'https://image.yes24.com/goods/145757238/XL',
    '브랜드 설계자': 'https://image.yes24.com/goods/120242691/XL',
    '24시간 완성! 챗GPT 스타트업 프롬프트 설계': 'https://image.yes24.com/goods/142637189/XL',
    '투자자는 무엇에 꽂히는가': 'https://image.yes24.com/goods/150108736/XL',
    '스토리 설계자': 'https://image.yes24.com/goods/130167416/XL',
    '스타트업 30분 회계': 'https://image.yes24.com/goods/148063482/XL',
    '세균무기의 스타트업 바운스백': 'https://image.yes24.com/goods/147976182/XL',
    'VC 스타트업': 'https://image.yes24.com/goods/125313295/XL',
    '스타트업 HR 팀장들': 'https://image.yes24.com/goods/126338963/XL',
    '스타트업 자금조달 바이블': 'https://image.yes24.com/goods/123878435/XL',
    '스타트업 디자인 씽킹': 'https://image.yes24.com/goods/116605554/XL'
}

# --- Session State 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.context = {"stage": None, "challenge": None}
    st.session_state.step = "stage_selection"

# ==============================================================================
# 모든 함수 정의
# ==============================================================================

# --- LangChain 체인(Chain) 정의 ---
def format_docs(docs):
    """검색된 문서들을 프롬프트에 넣기 좋은 형태로 포맷합니다."""
    return "\n\n".join(f"- **{doc.metadata['name']}** (저자: {doc.metadata['author']}): {doc.metadata['intro']}" for doc in docs)

def create_augmented_query(inputs):
    """사용자 정보를 포함하여 검색 쿼리를 강화합니다."""
    return f"성장 단계: {inputs['stage']}, 당면 과제: {inputs['challenge']}, 구체적인 고민: {inputs['user_prompt']}"

# 1. 질문 의도 분류 체인 (Router)
routing_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 질문 의도를 분석하는 AI입니다.
    사용자의 마지막 질문이 '새로운 주제의 책 추천'을 요구하는 것인지, 아니면 '이전에 추천된 책에 대한 추가 질문'인지 판단하세요.

    [대화 기록]
    {conversation_history}

    [사용자의 마지막 질문]
    "{user_prompt}"

    [판단 기준]
    - "다른 책 추천해줘", "마케팅 관련 책 찾아줘", "~에 대한 책 있어?" 와 같이 새로운 주제가 명시되면 'new_recommendation'입니다.
    - "그 책에 대해 더 알려줘", "저자가 누구야?", "MVP가 뭐야?" 와 같이 이전 대화의 대상을 가리키는 대명사가 있거나, 대화의 맥락을 이어가는 질문이면 'deepen_discussion'입니다.

    [출력]
    'new_recommendation' 또는 'deepen_discussion' 중 하나로만 답변하세요.
    """
)
routing_chain = routing_prompt | llm | StrOutputParser()

# 2. 책 추천 체인 (RAG)
recommendation_template = """
당신은 스타트업 창업가를 돕는 세계 최고의 컨설턴트입니다.
[사용자 정보]
- 성장 단계: '{stage}'
- 당면 과제: '{challenge}'
- 구체적인 고민: "{user_prompt}"
[1차 분석 결과: 가장 관련성 높은 책 후보 목록]
{retrieved_books}
[최종 미션]
1. [순위 결정] '후보 목록' 중에서 사용자의 '구체적인 고민'을 해결하는 데 가장 적합한 책을 1, 2, 3순위로 결정하세요. 책 제목은 후보 목록에 있는 것과 정확히 일치해야 합니다.
2. [근거 강화 리서치] 1순위 책에 대해, 추천의 신뢰도를 높일 **구체적인 근거**를 제시하세요. 'OOO 스타트업' 같은 모호한 표현은 절대 사용하지 마세요. 실제 사례를 찾기 어렵다면, 저자의 다른 아티클, 유명인의 긍정적인 리뷰 등을 **가상의 웹 검색 결과**처럼 만들어 근거로 제시하세요.
3. [목차 검색] 1순위 책의 **실제 목차**를 가상의 웹 검색을 통해 찾아서, 내용에 맞게 줄바꿈(\\n)을 포함한 텍스트로 정리해주세요.
4. [적용 방향 제안] 검색한 목차를 참고하여, 사용자가 자신의 스타트업에 **어떻게 적용해볼 수 있을지** 구체적인 예시를 2~3가지 제안해주세요. **반드시 각 제안을 "1. ", "2. " 와 같이 숫자로 시작하고 줄바꿈(\\n)으로 구분된 명확한 리스트 형식으로 작성해야 합니다.**
5. [후속 질문 생성] 사용자가 이어서 궁금해할 만한 **후속 질문 3개**를 생성해주세요.
6. [최종 답변 생성] 위의 모든 정보를 종합하여, 아래 JSON 형식에 맞춰 최종 답변을 생성하세요.
```json
{{
  "best_book": {{"title": "<1순위 책 제목>", "author": "<1순위 책 저자>"}},
  "new_reason": "<근거 강화 리서치 결과를 포함한, 새롭게 생성된 맞춤 추천 이유>",
  "table_of_contents": "<검색으로 찾은, 줄바꿈으로 정리된 목차 텍스트>",
  "application_points": "<숫자 리스트 형식으로 작성된 구체적인 적용 방향 제안>",
  "second_and_third_books": [
    {{"title": "<2순위 책 제목>", "author": "<2순위 책 저자>"}},
    {{"title": "<3순위 책 제목>", "author": "<3순위 책 저자>"}}
  ],
  "follow_up_questions": ["<생성된 후속 질문 1>", "<생성된 후속 질문 2>", "<생성된 후속 질문 3>"]
}}
```
"""
recommendation_prompt = ChatPromptTemplate.from_template(recommendation_template)
recommendation_parser = JsonOutputParser()

recommendation_chain = (
    {
        "retrieved_books": RunnableLambda(create_augmented_query) | retriever | RunnableLambda(format_docs),
        "user_prompt": lambda x: x["user_prompt"],
        "stage": lambda x: x["stage"],
        "challenge": lambda x: x["challenge"],
    }
    | recommendation_prompt
    | llm
    | recommendation_parser
)

# 3. 심화 질문 답변 체인
deepen_template = """
당신은 스타트업 전문 컨설턴트입니다. 아래는 사용자와의 이전 대화 기록입니다.
[대화 기록]
{conversation_history}
[사용자의 새로운 질문]
"{user_prompt}"
[미션]
대화의 맥락을 완벽하게 파악한 후, 사용자의 '새로운 질문'에 대해 웹 검색을 통해 찾은 정보처럼 구체적이고 깊이 있는 답변을 해주세요.
"""
deepen_prompt = ChatPromptTemplate.from_template(deepen_template)
deepen_chain = deepen_prompt | llm | StrOutputParser()

# --- 추천 결과 표시 함수 ---
def display_recommendation(reco, message_index):
    """AI의 추천 결과를 UI에 예쁘게 표시합니다."""
    if not isinstance(reco, dict) or "best_book" not in reco:
        st.error("AI로부터 유효한 답변을 받지 못했습니다. 잠시 후 다시 시도해주세요.")
        st.write("받은 답변:", reco)
        return

    best_book_info = reco.get('best_book', {})
    best_book_title = best_book_info.get('title')

    if not best_book_title:
        st.error("AI가 적절한 책을 찾지 못했습니다. 질문을 좀 더 구체적으로 해주시겠어요?")
        return

    filtered_df = all_books_df[all_books_df['name'] == best_book_title]
    if filtered_df.empty:
        st.error(f"AI가 추천한 '{best_book_title}' 책을 데이터베이스에서 찾을 수 없습니다.")
        return
    best_book_details = filtered_df.iloc[0]

    st.success("AI가 당신의 고민을 위해 고른 맞춤 추천 도서입니다!")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(COVER_IMAGES.get(best_book_title, "https://placehold.co/150x220/EEE/31343C?text=No+Cover"), use_container_width=True)
    with col2:
        st.subheader(f"📖 {best_book_title}")
        st.markdown(f"**저자:** {best_book_info.get('author')}")

    st.markdown("#### 🤔 AI의 맞춤 추천 이유")
    st.info(reco.get('new_reason'))

    st.markdown("#### 💡 이 책의 적용 방향 제안")
    st.warning(reco.get('application_points'))

    with st.expander("추천 도서 책소개 및 목차 보기"):
        st.markdown("##### **책 소개**")
        intro_text = best_book_details.get('intro', '소개 정보 없음')
        st.write(intro_text)
        st.markdown("---")
        st.markdown("##### **목차**")
        table_text = reco.get('table_of_contents', '목차 정보를 불러오지 못했습니다.')
        st.text(table_text)

    other_books = reco.get('second_and_third_books', [])
    if other_books:
        st.markdown("---")
        st.markdown("##### 📚 함께 읽으면 좋은 책들")
        cols = st.columns(len(other_books))
        for i, book in enumerate(other_books):
            with cols[i]:
                book_title = book.get('title')
                st.image(COVER_IMAGES.get(book_title, "https://placehold.co/100x150/EEE/31343C?text=No+Cover"), use_container_width=True)
                st.write(f"**{book_title}**")
                st.caption(f"_{book.get('author')}_")

    follow_up_questions = reco.get('follow_up_questions', [])
    if follow_up_questions:
        st.markdown("---")
        st.markdown("##### 💬 추가로 질문해 보세요")
        for i, question in enumerate(follow_up_questions):
            if st.button(question, key=f"follow_up_{message_index}_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.step = "generating_response"
                st.rerun()

# --- 대화 기록 가공 함수 ---
def create_conversation_history_string(messages):
    """대화 기록(messages)을 LLM이 이해하기 쉬운 자연어 문자열로 변환합니다."""
    history = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            history.append(f"사용자: {content}")
        elif role == "assistant":
            if isinstance(content, dict):
                book_title = content.get('best_book', {}).get('title', '알 수 없는 책')
                summary = f"AI: '{book_title}' 책을 포함한 관련 정보를 추천했습니다."
                history.append(summary)
            else:
                history.append(f"AI: {content}")
    return "\n".join(history)


# --- 메인 로직 (수정됨) ---
def handle_chat(prompt, is_initial_question=False):
    """사용자 입력을 받아 적절한 체인을 호출하고 AI 응답을 생성합니다."""
    response_content = None
    try:
        # 첫 질문일 경우, 무조건 추천 체인 실행
        if is_initial_question:
            response_content = recommendation_chain.invoke({
                "user_prompt": prompt,
                "stage": st.session_state.context['stage'],
                "challenge": st.session_state.context['challenge']
            })
        # 후속 질문일 경우, 의도 파악 후 분기
        else:
            history_for_llm = create_conversation_history_string(st.session_state.messages[:-1])
            intent = routing_chain.invoke({"conversation_history": history_for_llm, "user_prompt": prompt})

            # "새로운 추천" 요청 시, '당면 과제'를 현재 프롬프트로 덮어쓰고 추천 체인 실행
            if "new_recommendation" in intent:
                response_content = recommendation_chain.invoke({
                    "user_prompt": prompt,
                    "stage": st.session_state.context['stage'],
                    "challenge": prompt  # [수정] 새로운 질문을 새로운 과제로 사용
                })
            # "심화 질문" 시, 가공된 대화 기록을 포함하여 심화 답변 체인 실행
            else: # deepen_discussion
                response_content = deepen_chain.invoke({
                    "conversation_history": history_for_llm,
                    "user_prompt": prompt
                })

        st.session_state.messages.append({"role": "assistant", "content": response_content})

    except Exception as e:
        st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
        if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
            st.session_state.messages.pop()


# ==============================================================================
# 메인 UI 렌더링 및 단계별 로직
# ==============================================================================

st.title("🧭 스타트업 네비게이터")
st.caption("🚀 당신의 고민에 딱 맞는 책을 AI가 찾아드립니다!")

# --- 대화 기록 표시 ---
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        if i > 0:
            st.markdown("""
            <br>
            <hr style="margin-top: 20px; margin-bottom: 20px;">
            <br>
            """, unsafe_allow_html=True)
        st.markdown(f'<div class="user-message-container"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)

    elif message["role"] == "assistant":
        if isinstance(message["content"], dict):
            display_recommendation(message["content"], message_index=i)
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

# --- 단계별 UI 및 입력 처리 ---
if st.session_state.step == "stage_selection":
    st.info("안녕하세요! 어떤 고민을 가지고 계신가요? 먼저 현재 비즈니스의 성장 단계를 선택해주세요.")
    stages = ["아이디어 검증", "MVP 개발/초기 고객 확보", "PMF(시장-제품 적합성) 탐색", "스케일업/투자 유치"]
    for stage in stages:
        if st.button(stage, use_container_width=True):
            st.session_state.context["stage"] = stage
            st.session_state.step = "challenge_selection"
            st.rerun()

elif st.session_state.step == "challenge_selection":
    st.info(f"네, '{st.session_state.context['stage']}' 단계를 선택하셨군요. 이제 가장 고민되는 당면 과제를 선택해 주세요.")
    challenges = ["비즈니스 모델/전략", "제품/기술", "마케팅/영업", "팀/조직문화", "투자/재무"]
    for challenge in challenges:
        if st.button(challenge, use_container_width=True):
            st.session_state.context["challenge"] = challenge
            st.session_state.step = "problem_input"
            st.rerun()

elif st.session_state.step == "problem_input":
    st.info(f"'{st.session_state.context['challenge']}' 과제를 선택하셨군요. 이제 아래 채팅창에 구체적인 고민을 자유롭게 입력해주세요.")
    if prompt := st.chat_input("여기에 첫 고민을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.step = "generating_response"
        st.rerun()

elif st.session_state.step == "generating_response":
    with st.spinner("AI가 답변을 생성 중입니다... 잠시만 기다려주세요."):
        last_prompt = st.session_state.messages[-1]["content"]
        is_initial = sum(1 for msg in st.session_state.messages if msg['role'] == 'user') == 1
        handle_chat(last_prompt, is_initial_question=is_initial)
    st.session_state.step = "chat_mode"
    st.rerun()

elif st.session_state.step == "chat_mode":
    if prompt := st.chat_input("추가적인 질문을 입력하거나, 아래 추천 질문을 눌러보세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.step = "generating_response"
        st.rerun()
