# from dotenv import load_dotenv
# from pathlib import Path
import os
import base64
import io

# 환경 변수 설정 부분 제거 (웹 배포를 위해)
# 이제 사용자가 직접 API 키를 입력하게 됩니다

import streamlit as st
import json
import hashlib
import datetime
import re
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



# 페이지 구성 설정.
st.set_page_config(layout="wide", page_title="Prompt Editor")

# 세션 상태 초기화
def init_session_state():
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "current_version" not in st.session_state:
        st.session_state.current_version = 0
    if "versions" not in st.session_state:
        st.session_state.versions = []
    if "prompt_name" not in st.session_state:
        st.session_state.prompt_name = "Untitled Prompt"
    if "saved_prompts" not in st.session_state:
        st.session_state.saved_prompts = {}
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    if "current_prompt_id" not in st.session_state:
        st.session_state.current_prompt_id = None
    if "variables" not in st.session_state:
        st.session_state.variables = {}
    if "template_user_prompt" not in st.session_state:
        st.session_state.template_user_prompt = ""
    if "show_variables" not in st.session_state:
        st.session_state.show_variables = False
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gemini-2.0-flash"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.5
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 4000
    # API 키 관련 상태 추가
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = ""
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""
    # 선택된 모델 제공업체
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = None
    # 앱 상태 (설정 or 사용 모드)
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "setup"  # 초기값은 "setup", 키 설정 후 "editor"로 변경

# 세션 상태 초기화 함수 호출
init_session_state()

# 프롬프트 해시 생성 함수
def generate_hash(system_prompt, user_prompt):
    # template_user_prompt를 사용하여 해시 생성 (실제 변수 값이 적용된 user_prompt 대신)
    # 이렇게 하면 변수 값만 바뀌었을 때는 해시가 변경되지 않음
    combined = system_prompt + st.session_state.template_user_prompt
    return hashlib.md5(combined.encode()).hexdigest()

# 버전 저장 함수
def save_version(system_prompt, user_prompt, template_user_prompt=None):
    if template_user_prompt is None:
        template_user_prompt = st.session_state.template_user_prompt or user_prompt
        
    # 시스템 프롬프트와 템플릿 유저 프롬프트를 기준으로 해시 생성
    # 변수 값이 적용된 user_prompt는 해시 생성에 사용하지 않음
    current_hash = generate_hash(system_prompt, template_user_prompt)
    
    # 새 버전인지 확인 (해시가 다른 경우)
    if not st.session_state.versions or current_hash != st.session_state.versions[-1]["hash"]:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_version = {
            "version": len(st.session_state.versions) + 1,
            "timestamp": timestamp,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "template_user_prompt": template_user_prompt,
            "hash": current_hash
        }
        st.session_state.versions.append(new_version)
        st.session_state.current_version = len(st.session_state.versions)
        return True
    return False

# 버전 로드 함수
def load_version(version_index):
    if 0 <= version_index < len(st.session_state.versions):
        version = st.session_state.versions[version_index]
        st.session_state.system_prompt = version["system_prompt"]
        st.session_state.user_prompt = version["user_prompt"]
        
        # 버전에 템플릿이 있으면 로드
        if "template_user_prompt" in version:
            st.session_state.template_user_prompt = version["template_user_prompt"]
        else:
            # 템플릿이 없으면 사용자 프롬프트를 템플릿으로 사용
            st.session_state.template_user_prompt = version["user_prompt"]
        
        st.session_state.current_version = version_index + 1
        return True
    return False

# 프롬프트 파일 저장 함수
def save_prompt_to_file():
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    
    # 프롬프트에 고유 ID가 없으면 생성
    if not st.session_state.current_prompt_id:
        st.session_state.current_prompt_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    prompt_data = {
        "name": st.session_state.prompt_name,
        "versions": st.session_state.versions,
        "current_version": st.session_state.current_version,
        "prompt_id": st.session_state.current_prompt_id,
        "template_user_prompt": st.session_state.template_user_prompt,
        "variables": st.session_state.variables,
        "system_prompt": st.session_state.system_prompt
    }
    
    # 파일명에 프롬프트 ID를 포함하여 고유성 보장
    filename = f"prompts/{st.session_state.current_prompt_id}_{st.session_state.prompt_name.replace(' ', '_')}.json"
    with open(filename, "w") as f:
        json.dump(prompt_data, f, indent=2)


    # # 파일명에 프롬프트 ID를 포함하여 고유성 보장
    # filename = f"prompts/{st.session_state.current_prompt_id}_{st.session_state.prompt_name.replace(' ', '_')}.json"
    # with open(filename, "w", encoding="utf-8") as f:
    #     json.dump(prompt_data, f, indent=2, ensure_ascii=False)

    
    # 저장된 프롬프트 목록 업데이트
    st.session_state.saved_prompts[st.session_state.prompt_name] = filename
    return filename

# 프롬프트 파일 로드 함수
def load_prompt_from_file(prompt_name):
    filename = st.session_state.saved_prompts.get(prompt_name)
    if filename and os.path.exists(filename):
        with open(filename, "r") as f:
            prompt_data = json.load(f)
        
        # 로드하기 전에 현재 버전 기록 초기화
        st.session_state.versions = []
        
        st.session_state.prompt_name = prompt_data["name"]
        st.session_state.versions = prompt_data["versions"]
        st.session_state.current_version = prompt_data["current_version"]
        
        # 현재 프롬프트 ID 설정
        if "prompt_id" in prompt_data:
            st.session_state.current_prompt_id = prompt_data["prompt_id"]
        else:
            # 이전에 저장된 프롬프트용 ID 생성
            st.session_state.current_prompt_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 템플릿과 변수 로드 (있는 경우)
        if "template_user_prompt" in prompt_data:
            st.session_state.template_user_prompt = prompt_data["template_user_prompt"]
        else:
            st.session_state.template_user_prompt = ""
            
        # 변수 초기화 (빈 딕셔너리로 설정)
        # 저장된 변수 값을 로드하지 않고 빈 값으로 시작
        st.session_state.variables = {}
        
        # 시스템 프롬프트 로드 (있는 경우)
        if "system_prompt" in prompt_data:
            st.session_state.system_prompt = prompt_data["system_prompt"]
        else:
            st.session_state.system_prompt = ""
        
        # 현재 버전 로드
        current_idx = st.session_state.current_version - 1
        if 0 <= current_idx < len(st.session_state.versions):
            version = st.session_state.versions[current_idx]
            st.session_state.system_prompt = version["system_prompt"]
            st.session_state.user_prompt = version["user_prompt"]
        
        return True
    return False

# 기존 프롬프트 검색 함수
def scan_for_prompts():
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    
    saved_prompts = {}
    for filename in os.listdir("prompts"):
        if filename.endswith(".json"):
            try:
                with open(os.path.join("prompts", filename), "r") as f:
                    prompt_data = json.load(f)
                prompt_name = prompt_data.get("name", filename[:-5].replace("_", " "))
                saved_prompts[prompt_name] = os.path.join("prompts", filename)
            except:
                pass
    
    st.session_state.saved_prompts = saved_prompts
    
    # 새 세션을 위한 프롬프트 ID 생성
    if st.session_state.current_prompt_id is None:
        st.session_state.current_prompt_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# 시작 시 기존 프롬프트 검색
if not st.session_state.saved_prompts:
    scan_for_prompts()

# 프롬프트에서 변수 추출 함수
def extract_variables(prompt):
    variables = []
    
    # {variable} 및 {{variable}} 패턴 모두 매칭
    patterns = [r'\{([^{}]+)\}', r'\{\{([^{}]+)\}\}']
    
    for pattern in patterns:
        matches = re.findall(pattern, prompt)
        variables.extend(matches)
    
    # 중복 제거하면서 순서 유지
    unique_vars = []
    for var in variables:
        if var not in unique_vars:
            unique_vars.append(var)
    
    return unique_vars

# 템플릿에 변수 적용 함수
def apply_variables(template, variables_dict):
    result = template
    for var_name, var_value in variables_dict.items():
        # {var} 및 {{var}} 패턴 모두 대체
        result = result.replace('{{' + var_name + '}}', var_value)
        result = result.replace('{' + var_name + '}', var_value)
    return result

# LLM 체인 초기화 함수
def get_llm_chain(model_name, temperature, max_tokens):
    if model_name.startswith("gpt"):
        llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
    elif model_name.startswith("claude"):
        llm = ChatAnthropic(
            api_key=st.session_state.anthropic_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
    elif model_name.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(          
            google_api_key=st.session_state.gemini_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            # streaming=True
        )
    else:
        raise ValueError(f"지원되지 않는 모델: {model_name}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "{human}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain

# 응답 생성 함수
def generate_response(system_prompt, user_prompt, model_name, temperature, max_tokens):
    try:
        # API 키 확인
        if model_name.startswith("gpt") and not st.session_state.openai_api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. API 키를 설정해주세요.")
            return
        elif model_name.startswith("claude") and not st.session_state.anthropic_api_key:
            st.error("Anthropic API 키가 설정되지 않았습니다. API 키를 설정해주세요.")
            return
        elif model_name.startswith("gemini") and not st.session_state.gemini_api_key:
            st.error("Google Gemini API 키가 설정되지 않았습니다. API 키를 설정해주세요.")
            return
            
        chain = get_llm_chain(model_name, temperature, max_tokens)
        
        # 응답 초기화
        st.session_state.response = ""
        
        # 글로벌 변수로 응답 플레이스홀더 접근
        global response_placeholder
        
        # 응답 스트리밍
        for chunk in chain.stream({"system": system_prompt, "human": user_prompt}):
            st.session_state.response += chunk
            # 응답 표시
            if 'response_placeholder' in globals():
                response_placeholder.markdown(st.session_state.response)
    except Exception as e:
        st.error(f"응답 생성 오류: {str(e)}")

# 앱 제목
st.title("Prompt Editor")
st.markdown("© 2025 cogdex | cogPrompt™")

# 앱 모드에 따라 다른 화면 표시 (setup 또는 editor)
if st.session_state.app_mode == "setup":
    st.header("API 키 설정")
    st.markdown("프롬프트 에디터를 사용하기 위해 아래에서 사용할 AI 모델 제공업체를 선택하고 해당 API 키를 입력해주세요.")
    
    # 모델 제공업체 선택
    provider_options = ["OpenAI", "Anthropic", "Google Gemini"]
    selected_provider = st.selectbox(
        "모델 제공업체 선택",
        options=provider_options,
        index=provider_options.index(st.session_state.selected_provider) if st.session_state.selected_provider in provider_options else 0
    )
    st.session_state.selected_provider = selected_provider
    
    # 선택된 제공업체에 따라 API 키 입력 필드 표시
    api_key = ""
    if selected_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        st.session_state.openai_api_key = api_key
    elif selected_provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", value=st.session_state.anthropic_api_key, type="password")
        st.session_state.anthropic_api_key = api_key
    elif selected_provider == "Google Gemini":
        api_key = st.text_input("Google Gemini API Key", value=st.session_state.gemini_api_key, type="password")
        st.session_state.gemini_api_key = api_key
    
    # 시작 버튼
    if st.button("시작하기", key="start_button"):
        if api_key:
            # 모델 선택 기본값 설정
            if selected_provider == "OpenAI":
                st.session_state.model_name = "gpt-3.5-turbo"
            elif selected_provider == "Anthropic":
                st.session_state.model_name = "claude-3-5-sonnet-20241022"
            elif selected_provider == "Google Gemini":
                st.session_state.model_name = "gemini-2.0-flash"
            
            st.session_state.app_mode = "editor"
            st.rerun()
        else:
            st.error("API 키를 입력해주세요.")
else:
    # 기존 에디터 UI 유지
    # 사이드바에 프롬프트 관리, 모델 설정 등을 배치
    with st.sidebar:
        # API 키 변경 버튼 추가
        if st.button("API 키 변경", key="change_api_key"):
            st.session_state.app_mode = "setup"
            st.rerun()
            
        st.header("Prompt Management")
        
        # 프롬프트 이름 및 저장 기능 (같은 행에 배치)
        name_col, save_col = st.columns([3, 1])
        with name_col:
            new_prompt_name = st.text_input("Enter Prompt Name", value=st.session_state.prompt_name, key="prompt_name_input")
            if new_prompt_name != st.session_state.prompt_name:
                st.session_state.prompt_name = new_prompt_name
                # 완전히 새로운 프롬프트 생성 시 버전 기록 초기화
                if not any(name.lower() == new_prompt_name.lower() for name in st.session_state.saved_prompts.keys()):
                    st.session_state.versions = []
                    st.session_state.current_version = 0
        
        with save_col:
            st.write("") # 레이블과 버튼 높이를 맞추기 위한 공백
            st.write("")
            save_button_clicked = st.button("Save", key="save_button")
        
        if save_button_clicked:
            filename = save_prompt_to_file()
            st.success(f"Saved as {filename}")
        
        # 불러오기 옵션과 로드 버튼 (같은 행에 배치)
        prompt_options = list(st.session_state.saved_prompts.keys())
        if prompt_options:
            load_col, load_button_col = st.columns([3, 1])
            with load_col:
                selected_prompt = st.selectbox("Load Prompt", options=prompt_options, key="load_select")
            with load_button_col:
                st.write("") # 레이블과 버튼 높이를 맞추기 위한 공백
                st.write("")
                load_button_clicked = st.button("Load", key="load_button")

            # 버튼 클릭 처리를 컬럼 바깥에서 수행
            if load_button_clicked:
                if load_prompt_from_file(selected_prompt):
                    st.success(f"Loaded: {selected_prompt}")
                else:
                    st.error("Failed to load prompt")

        # 다운로드/업로드 기능 추가
        st.write("---")
        st.subheader("Import/Export Prompt")
        
        # 다운로드 기능
        if st.button("📥 Download Current Prompt"):
            # 현재 프롬프트 데이터 생성
            prompt_data = {
                "name": st.session_state.prompt_name,
                "versions": st.session_state.versions,
                "current_version": st.session_state.current_version,
                "prompt_id": st.session_state.current_prompt_id or datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                "template_user_prompt": st.session_state.template_user_prompt,
                "variables": {},  # 변수 값은 저장하지 않음
                "system_prompt": st.session_state.system_prompt
            }
            
            # JSON 파일로 변환
            json_str = json.dumps(prompt_data, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            
            # 다운로드 링크 생성
            filename = f"{st.session_state.prompt_name.replace(' ', '_')}.json"
            href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Click to download {filename}</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # 업로드된 파일 처리 함수 정의
        def process_uploaded_file():
            if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
                try:
                    # 업로드된 파일 내용 읽기
                    content = st.session_state.uploaded_file.read()
                    prompt_data = json.loads(content.decode())
                    
                    # 세션 상태 업데이트
                    st.session_state.prompt_name = prompt_data["name"]
                    st.session_state.versions = prompt_data["versions"]
                    st.session_state.current_version = prompt_data["current_version"]
                    
                    # 프롬프트 ID 설정
                    if "prompt_id" in prompt_data:
                        st.session_state.current_prompt_id = prompt_data["prompt_id"]
                    else:
                        # 이전에 저장된 프롬프트용 ID 생성
                        st.session_state.current_prompt_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    
                    # 템플릿 로드
                    if "template_user_prompt" in prompt_data:
                        st.session_state.template_user_prompt = prompt_data["template_user_prompt"]
                    else:
                        st.session_state.template_user_prompt = ""
                    
                    # 변수는 항상 빈 상태로 초기화
                    st.session_state.variables = {}
                    
                    # 시스템 프롬프트 로드
                    if "system_prompt" in prompt_data:
                        st.session_state.system_prompt = prompt_data["system_prompt"]
                    else:
                        st.session_state.system_prompt = ""
                    
                    # 현재 버전 로드
                    current_idx = st.session_state.current_version - 1
                    if st.session_state.versions and 0 <= current_idx < len(st.session_state.versions):
                        version = st.session_state.versions[current_idx]
                        st.session_state.system_prompt = version["system_prompt"]
                        st.session_state.user_prompt = version["user_prompt"]
                        if "template_user_prompt" in version:
                            st.session_state.template_user_prompt = version["template_user_prompt"]
                    
                    st.success(f"Prompt '{st.session_state.prompt_name}' successfully uploaded!")
                    
                    # 파일 업로더 상태 초기화
                    st.session_state.uploaded_file = None
                    
                except Exception as e:
                    st.error(f"Error uploading prompt file: {str(e)}")
        
        # 파일이 업로드되면 세션 상태에 저장
        def on_file_upload():
            st.session_state.uploaded_file = uploaded_file
        
        # 업로드 기능
        uploaded_file = st.file_uploader("📤 Upload Prompt File", type=["json"], on_change=on_file_upload)
        
        # 업로드된 파일 처리
        process_uploaded_file()

        # 새 프롬프트 초기화 버튼
        if st.button("🔄 :gray[**Create New Prompt**]", key="new_prompt_button"):
            # 새로고침과 비슷하게 세션 상태 초기화
            st.session_state.system_prompt = ""
            st.session_state.user_prompt = ""
            st.session_state.response = ""
            st.session_state.current_version = 0
            st.session_state.versions = []
            st.session_state.prompt_name = "Untitled Prompt"
            st.session_state.current_prompt_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            st.session_state.variables = {}
            st.session_state.template_user_prompt = ""
            st.session_state.show_variables = False
            # 새로고침 효과를 위해 리로드
            st.rerun()
        
        st.write("---")

        # 모델 파라미터 설정
        st.header("Model Parameters")
        
        # 선택된 제공업체에 따라 모델 옵션 필터링
        if st.session_state.selected_provider == "OpenAI":
            model_options = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
            ]
        elif st.session_state.selected_provider == "Anthropic":
            model_options = [
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-haiku-20241022",
                "claude-3-haiku-20240307",
            ]
        elif st.session_state.selected_provider == "Google Gemini":
            model_options = [
                "gemini-2.0-flash",
                "gemini-2.5-pro-exp-03-25",
            ]
        else:
            # 기본 옵션
            model_options = [
                "gemini-2.0-flash",
                "claude-3-5-sonnet-20241022",
                "gpt-3.5-turbo",
            ]
            
        # 현재 선택된 모델이 옵션에 없으면 첫 번째 모델 선택
        default_index = 0
        if st.session_state.model_name in model_options:
            default_index = model_options.index(st.session_state.model_name)
            
        selected_model = st.selectbox(
            "Select Model", 
            options=model_options, 
            index=default_index, 
            key="model_select"
        )
        st.session_state.model_name = selected_model
        
        selected_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1, key="temp_slider")
        st.session_state.temperature = selected_temp
        
        selected_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=st.session_state.max_tokens, step=100, key="tokens_input")
        st.session_state.max_tokens = selected_tokens
        
        # 응답 표시를 위한 전역 플레이스홀더 선언
        global response_placeholder
        
        st.write("---")
        
        # 디버그 정보 표시 (접을 수 있는 섹션으로 제공)
        with st.expander("Show Debug Info", expanded=False):
            st.write("Detected Variables:", extract_variables(st.session_state.template_user_prompt))
            st.write("Current Prompt ID:", st.session_state.current_prompt_id)
            st.write("Current Version:", st.session_state.current_version)
            st.write("Number of Versions:", len(st.session_state.versions))
            st.write("Model:", st.session_state.model_name)
            st.write("Temperature:", st.session_state.temperature)
            st.write("Max Tokens:", st.session_state.max_tokens)
            st.write("Selected Provider:", st.session_state.selected_provider)

# 메인 영역에는 프롬프트 입력과 응답 표시 영역을 배치
# 왼쪽 열에는 프롬프트 입력, 오른쪽 열에는 응답 표시
left_col, right_col = st.columns([1, 1])

# 왼쪽 열: 프롬프트 입력 영역을 세로로 배치
with left_col:
    # 세로 분할을 위한 컨테이너 사용
    system_container = st.container()
    user_container = st.container()
    
    # 시스템 프롬프트 (상단)
    with system_container:
        # st.subheader("System Prompt")
        st.markdown("#### System Prompt")
        system_prompt = st.text_area("Set a system instruction", value=st.session_state.system_prompt, height=100, key="system_prompt_area")
        st.session_state.system_prompt = system_prompt
    
    # 사용자 프롬프트 (하단)
    with user_container:
        # st.subheader("User")
        st.markdown("#### User")
        template_user_prompt = st.text_area("Enter user message: 변수는 {variable} 또는 {{variable}} 형식으로 사용", 
                                          value=st.session_state.template_user_prompt or st.session_state.user_prompt, 
                                          height=250, 
                                          key="user_prompt_area")
        
        # 템플릿 업데이트
        st.session_state.template_user_prompt = template_user_prompt
        
        # 변수 토글 버튼
        if st.button(("🏷️ Hide" if st.session_state.show_variables else "🏷️ Show") + " :orange[**Variables**]", key="variables_toggle_main"):
            st.session_state.show_variables = not st.session_state.show_variables
            
        # Run 버튼 추가 (Show Variables 아래에 배치)
        if st.button("▶️ :blue[**Run**]", key="run_button"):
            # 변경 사항 있으면 버전 저장
            is_new_version = save_version(st.session_state.system_prompt, st.session_state.user_prompt, st.session_state.template_user_prompt)
            if is_new_version:
                st.success(f"새 버전 {len(st.session_state.versions)} 생성됨")
                
            # 응답 탭으로 전환하도록 안내
            st.success("응답이 'Response' 탭에 표시됩니다")
                
            # 응답 생성 (메인 영역의 응답 플레이스홀더에 표시됨)
            with st.spinner("Generating response..."):
                generate_response(
                    st.session_state.system_prompt,
                    st.session_state.user_prompt,
                    st.session_state.model_name,
                    st.session_state.temperature,
                    st.session_state.max_tokens
                )

        # 템플릿에서 변수 추출
        variables = extract_variables(template_user_prompt)
        
        # 변수 섹션 표시 (토글 시)
        if st.session_state.show_variables:
            if variables:
                # 변수 입력을 위한 컬럼 생성 (행당 2개 변수)
                cols_per_row = 2
                rows = (len(variables) + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        var_idx = row * cols_per_row + col_idx
                        if var_idx < len(variables):
                            var_name = variables[var_idx]
                            # 기존 값 가져오기 또는 빈 문자열
                            current_value = st.session_state.variables.get(var_name, "")
                            # 이 변수의 입력 필드 생성
                            var_value = cols[col_idx].text_input(f"{var_name}", value=current_value, key=f"var_{var_name}")
                            # 변수 값 저장
                            st.session_state.variables[var_name] = var_value
            else:
                st.info("템플릿에서 변수가 감지되지 않았습니다. {variable} 또는 {{variable}} 구문을 사용하여 변수를 추가하세요.")
        
        # 변수를 템플릿에 적용 (결과 프롬프트는 표시하지 않고 내부적으로만 처리)
        if variables:
            user_prompt = apply_variables(template_user_prompt, st.session_state.variables)
            # 변수가 적용된 사용자 프롬프트 업데이트
            st.session_state.user_prompt = user_prompt
        else:
            # 변수가 없으면 템플릿을 그대로 사용
            st.session_state.user_prompt = template_user_prompt

# 전역 변수로 응답 플레이스홀더 선언
response_placeholder = None



# 오른쪽 열: 응답 표시와 버전 히스토리
with right_col:
    # 탭을 사용하여 응답과 버전 히스토리를 구분
    response_tab, history_tab = st.tabs(["**Response**", "**Version History**"])
    
    # 응답 탭
    with response_tab:
        # 응답 표시를 위한 플레이스홀더
        response_placeholder = st.empty()
        
        # 저장된 응답 표시
        if st.session_state.response:
            response_placeholder.markdown(st.session_state.response)
    
    # 버전 히스토리 탭
    with history_tab:
        st.subheader("Version History")
        if st.session_state.versions:
            # 버전 히스토리를 스크롤 가능한 컨테이너에 배치
            with st.container():
                # 버전 정보를 테이블로 표시
                version_data = []
                for i, version in enumerate(st.session_state.versions):
                    is_current = i + 1 == st.session_state.current_version
                    version_data.append({
                        "Version": version['version'],
                        "Timestamp": version['timestamp'],
                        "Current": "✓" if is_current else "",
                        "Action": f"version_{i}"  # 버튼을 위한 키
                    })
                
                # 데이터프레임으로 버전 리스트 표시
                import pandas as pd
                df = pd.DataFrame(version_data)
                
                # 테이블 표시
                st.dataframe(df[["Version", "Timestamp", "Current"]], hide_index=True)
                
                # 버전 선택을 위한 두 개의 컬럼 생성 (선택 드롭다운과 로드 버튼)
                col1, col2 = st.columns([3, 1])
                with col1:
                    version_options = [f"Version {v['version']} ({v['timestamp']})" for v in st.session_state.versions]
                    selected_version_idx = st.selectbox(
                        "Select Version to Load",
                        options=range(len(version_options)),
                        format_func=lambda x: version_options[x],
                        index=st.session_state.current_version - 1 if st.session_state.current_version > 0 else 0
                    )
                with col2:
                    st.write("")  # 버튼 정렬을 위한 공백
                    st.write("")
                    if st.button("Load Version"):
                        if load_version(selected_version_idx):
                            st.success(f"버전 {selected_version_idx + 1} 로드 완료")
                
                # 버전 비교 기능 (선택적)
                with st.expander("Version Details", expanded=False):
                    if st.session_state.versions:
                        selected_version = st.session_state.versions[selected_version_idx]
                        st.write(f"**Version {selected_version['version']} Details:**")
                        st.write(f"Created: {selected_version['timestamp']}")
                        st.text_area("System Prompt", value=selected_version["system_prompt"], height=150, disabled=True)
                        st.text_area("User Prompt", value=selected_version["user_prompt"], height=150, disabled=True)
        else:
            st.info("아직 버전 히스토리가 없습니다. 변경 사항이 있으면 'Run' 버튼을 클릭하여 새 버전을 만드세요.")


