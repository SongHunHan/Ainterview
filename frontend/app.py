import streamlit as st
import requests

st.title("채용공고 기반 면접 질문 생성기")

# CSS를 사용하여 스크롤 가능한 텍스트 영역 스타일 정의
st.markdown("""
<style>
.scrollable-text {
    height: 150px;
    overflow-y: auto;
    border: 1px solid #444;
    padding: 10px;
    background-color: #1e1e1e;
    color: #ffffff;
    margin-bottom: 30px;  # 아래쪽 여백을 30px로 증가
}
.scrollable-text strong {
    color: #4da6ff;
}
.question-box {
    margin-bottom: 40px;  # 질문 박스 아래에 40px의 여백 추가
}
</style>
""", unsafe_allow_html=True)

job_post = st.text_area("채용공고 내용을 입력하세요", height=200)

# 세션 상태 초기화
if 'interviews' not in st.session_state:
    st.session_state.interviews = [{"question": "", "answer": ""} for _ in range(3)]

col1, col2, col3 = st.columns(3)

for i, col in enumerate([col1, col2, col3], start=1):
    with col:
        st.image(f"frontend/assets/question_image{i}.png", caption=f"면접관 {i}")
        
        if st.button(f"질문/답변 {i} 생성"):
            response = requests.post(f"http://localhost:8000/generate_interview/{i-1}", json={"content": job_post})
            if response.status_code == 200:
                data = response.json()
                st.session_state.interviews[i-1] = {
                    "question": data["question"],
                    "answer": data["answer"]
                }
            else:
                st.error("오류가 발생했습니다. 다시 시도해주세요.")

        # 질문 표시
        if st.session_state.interviews[i-1]["question"]:
            st.markdown(f"<div class='scrollable-text question-box'><strong>질문:</strong><br>{st.session_state.interviews[i-1]['question']}</div>", unsafe_allow_html=True)
        
        # 답변 표시
        if st.session_state.interviews[i-1]["answer"]:
            st.markdown(f"<div class='scrollable-text'><strong>답변:</strong><br>{st.session_state.interviews[i-1]['answer']}</div>", unsafe_allow_html=True)