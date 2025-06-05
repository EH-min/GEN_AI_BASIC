from backend.db.session import SessionLocal

def get_db():
	db = SessionLocal()
	try:
		# return: 모든 결과 값을 메모리에 올려 놓기
		# yield : 결과 값을 하나씩 메모리에 올려 놓기(속도 빠름, 제너레이터)
		yield db   
	finally:
		db.close()
		