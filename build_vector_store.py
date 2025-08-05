import pandas as pd
import os
from dotenv import load_dotenv
import time # [추가] API 요청 사이에 지연을 주기 위해 time 라이브러리를 가져옵니다.

# LangChain 관련 라이브러리들을 가져옵니다.
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def build_and_save_vector_store():
    """
    books_data_new.csv를 읽어 LangChain의 FAISS 벡터 저장소를 생성하고,
    'faiss_index' 폴더에 저장합니다.
    """
    # --- 1. API 키 로드 ---
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OpenAI API 키를 .env 파일에 설정해주세요.")

    # --- 2. 원본 데이터 로드 ---
    try:
        df = pd.read_csv('books_data_new.csv')
        df['intro'] = df['intro'].fillna("")
        df['table'] = df['table'].fillna("")
    except FileNotFoundError:
        print("오류: books_data_new.csv 파일을 찾을 수 없습니다.")
        return

    print("데이터를 LangChain 문서 형식으로 변환 중입니다...")
    
    # --- 3. LangChain 문서(Document) 형식으로 변환 ---
    documents = []
    for index, row in df.iterrows():
        page_content = f"책 소개: {row['intro']}\n\n목차: {row['table']}"
        metadata = {
            "name": row['name'],
            "author": row['author'],
            "intro": row['intro'],
            "table": row['table']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    # --- 4. 임베딩 모델 준비 ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("FAISS 벡터 저장소를 생성합니다. (API 요청량 제한으로 인해 시간이 걸릴 수 있습니다)")
    
    # --- [핵심 수정] 5. FAISS 벡터 저장소 '배치 처리'로 생성 및 저장 ---
    # 한 번에 모든 문서를 처리하지 않고, 작은 그룹으로 나누어 처리합니다.
    batch_size = 5  # 한 번에 5개씩 처리
    vector_store = None

    for i in range(0, len(documents), batch_size):
        batch_documents = documents[i : i + batch_size]
        print(f"{i+1}번째부터 {i+len(batch_documents)}번째 문서 처리 중...")
        
        if vector_store is None:
            # 첫 번째 배치로 벡터 저장소를 초기화합니다.
            vector_store = FAISS.from_documents(batch_documents, embeddings)
        else:
            # 이후 배치부터는 기존 저장소에 문서를 추가합니다.
            vector_store.add_documents(batch_documents)
        
        # API의 분당 요청량 제한(TPM)을 준수하기 위해 잠시 대기합니다.
        print("Rate Limit을 피하기 위해 20초 대기합니다...")
        time.sleep(20)

    # 생성된 인덱스를 'faiss_index'라는 폴더에 저장합니다.
    if vector_store:
        vector_store.save_local("faiss_index")
        print("✅ 'faiss_index' 폴더에 벡터 저장소가 성공적으로 저장되었습니다.")
    else:
        print("처리할 문서가 없어 벡터 저장소를 생성하지 못했습니다.")


if __name__ == "__main__":
    build_and_save_vector_store()
