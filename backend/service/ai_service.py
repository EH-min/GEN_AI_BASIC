import os
import time
import boto3
import requests

from io import BytesIO
from openai import InternalServerError, OpenAI
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI


# State 정의
class State(TypedDict):
    id: str
    prompt: str
    image_url: str
    save_path: str
    s3_url: str


class AIService:
    def __init__(self):
        _ = load_dotenv(find_dotenv())

        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)  # DALL·E-3 이미지 생성

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # 프롬프트 확장용 GPT

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-northeast-2")
        )
        self.BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

    def gen_graph(self, prompt: str):
        workflow = StateGraph(State)
        workflow.add_node("generate_image", self.generate_image)
        workflow.add_node("refine_prompt", self.refine_prompt)
        workflow.add_node("translate_prompt", self.translate_prompt)

        workflow.set_entry_point("refine_prompt")
        workflow.add_edge("refine_prompt", "translate_prompt")
        workflow.add_edge("translate_prompt", "generate_image")
        workflow.set_finish_point("generate_image")

        return workflow.compile()

    def save_image(self, image_url: str, file_name: str):
        response = requests.get(image_url)
        if response.status_code == 200:
            # S3에 업로드
            self.s3.upload_fileobj(BytesIO(response.content), self.BUCKET_NAME, file_name)
            s3_url = f"https://{self.BUCKET_NAME}.s3.amazonaws.com/{file_name}"
            print(f"S3 업로드 완료: {s3_url}")

            path = "/app/backend/downloads/"
            os.makedirs(path, exist_ok=True)

            save_path = f"{path}{file_name}"
            with open(save_path, 'wb') as f:
                f.write(response.content)

            print(f"이미지 저장 완료: {save_path}")
        else:
            raise Exception(f"이미지 다운로드 실패: {response.status_code}")

        return save_path, s3_url

    def refine_prompt(self, state: State):
        user_input = state["prompt"]
        response = self.llm.invoke(f"다음 문장을 이미지 생성용으로 개선하고, 개선한 문장만 출력해줘: {user_input}")
        print(f">> Prompt 확장: {response.content}")
        return {"prompt": response.content}

    def translate_prompt(self, state: State):
        user_input = state["prompt"]
        response = self.llm.invoke(f"다음 문장을 영어로 번역해줘: {user_input}")
        print(f">> Prompt 번역: {response.content}")
        return {"prompt": response.content}

    def generate_image(self, state: State) -> State:
        prompt = state["prompt"]
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            print(f"이미지 URL: {image_url}")

            file_name = state["id"] + ".png"
            save_path, s3_url = self.save_image(image_url, file_name)

            return {
                "prompt": prompt,
                "image_url": image_url,
                "save_path": save_path,
                "s3_url": s3_url
            }
        except InternalServerError as e:
            print(f"Exception: {e}")

        return {"prompt": prompt, "image_url": "error: openai server error after retries"}