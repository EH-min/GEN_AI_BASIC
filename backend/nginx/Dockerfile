FROM nginx:latest

RUN rm /etc/nginx/nginx.conf
RUN rm /etc/nginx/conf.d/default.conf

# 글로벌 설정
COPY nginx.conf /etc/nginx/ 
# 사이트/서비스별 설정
COPY project.conf /etc/nginx/conf.d/