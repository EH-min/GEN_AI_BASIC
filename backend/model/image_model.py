from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ImageORM(Base):
    __tablename__ = "image_prompts"
    
    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text, nullable=False) 
    save_path = Column(String, nullable=False) 
    s3_url = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False) 
    
    def __init__(self, id, prompt, save_path, s3_url, created_at):
        self.id = id
        self.prompt = prompt
        self.save_path = save_path
        self.s3_url = s3_url
        self.created_at = created_at
    
    def __repr__(self):
        return "<Image Prompt('%s', '%s', '%s')>" % (self.id, self.prompt, self.save_path, self.s3_url)