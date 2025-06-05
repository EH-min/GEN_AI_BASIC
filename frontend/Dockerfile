# Node.js 공식 이미지 사용
FROM node:18

# 작업 디렉토리 설정
WORKDIR /app/frontend

# 의존성 파일 복사
COPY ./frontend/package*.json ./

# 의존성 설치
RUN npm install

# 애플리케이션 소스 복사
COPY ./frontend /app/frontend

# [개발 모드]로 실행할 경우 (개발 서버)
CMD ["npm", "run", "dev", "--", "--host"]


# [배포용]
# RUN npm run build
# CMD ["node", "build"]

EXPOSE 5000