
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from app.models.base import Base


class Province(Base):
    __tablename__ = "provinces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    code = Column(String(10), unique=True, nullable=False) 
    main_city = Column(String(100))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    attractions = relationship("TouristAttraction", back_populates="province")
    





















