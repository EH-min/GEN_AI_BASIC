import os
import uuid
import requests
from typing_extensions import TypedDict
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from openai import OpenAI

class State(TypedDict):
    id: str # uuid
    prompt: str # 사용자 프롬프트
    image_url: str # 생성된 이미지 URL

class AIService:
  def __init__(self):
    _ = load_dotenv(find_dotenv())
    openai_api_key = os.getenv("OPENAI_API_KEY")
    self.llm = ChatOpenAI(
      model="gpt-4o-mini",
      temperature=0,
      openai_api_key=openai_api_key,
    )
    self.client = OpenAI(api_key=openai_api_key)

  def download_image(self, image_url: str, save_path: str) -> None:
    response = requests.get(image_url)
    if response.status_code == 200:
      with open(save_path, "wb") as f:
        f.write(response.content)
      print(f"이미지 저장 완료:  {save_path}")
    else:
      print(f"이미지 다운로드 실패: Status code: {response.status_code}")

  def refine_prompt(self, state: State) -> State:
    user_input = state["prompt"]
    response = self.llm.invoke(f"다음 문장을 이미지 생성용으로 개선하고, 개선한 문장만 출력해줘: {user_input}")
    print(f"확장된 프롬프트: {response.content}")
    return {"prompt": response.content}

  def translate_prompt(self, state: State) -> State:
    prompt = state["prompt"]
    response = self.llm.invoke(f"다음 문장을 영어로 번역해줘: {prompt}")
    print(f"번역 결과: {response.content}")
    return {"prompt": response.content}

  def generate_image(self, state: State) -> State:
    prompt = state["prompt"]
    response = self.client.images.generate(
      model="dall-e-3",
      prompt=prompt,
      size="1024x1024",
      quality="standard",
      n=1,
    )
    image_url = response.data[0].url
    save_path = f"./downloads/{uuid.uuid4()}.png"
    self.download_image(image_url, save_path)

    return {"prompt": prompt, "image_url": image_url}
  
  def gen_graph(self, prompt: str):
    workflow = StateGraph(State)
    workflow.add_node("refine_prompt", self.refine_prompt)
    workflow.add_node("translate_prompt", self.translate_prompt)
    workflow.add_node("generate_image", self.generate_image)

    workflow.set_entry_point("refine_prompt") # 시작 노드
    workflow.add_edge("refine_prompt", "translate_prompt") # 노드 연결
    workflow.add_edge("translate_prompt", "generate_image") # 노드 연결
    workflow.set_finish_point("generate_image") # 종료 노드

    graph = workflow.compile() # 그래프 컴파일
    return graph