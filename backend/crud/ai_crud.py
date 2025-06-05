from backend.schema.image_schema import ImageDTO
from backend.model.image_model import ImageORM
from sqlalchemy.orm import Session


def save_image_prompt(db: Session, image_dto: ImageDTO):
    db_image = ImageORM(
        id=image_dto.id,
        prompt=image_dto.prompt,
        save_path=image_dto.save_path,
        s3_url=image_dto.s3_url,
        created_at=image_dto.created_at
    )
    db.add(db_image)
    db.commit()
    