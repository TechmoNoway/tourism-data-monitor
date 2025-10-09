
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()

class Province(Base):
    __tablename__ = "provinces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    code = Column(String(10), unique=True, nullable=False) 
    main_city = Column(String(100))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    # relationships
    attractions = relationship("TouristAttraction", back_populates="province")
    





















