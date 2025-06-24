from typing import Dict, List, Optional
from pydantic import BaseModel, Field

TOTAL_CREATIONLOOK_PROMPT_ru = """\
Вы - профессиональный стилист, создающий total look.
Вот запрос пользователя: {request}.
Проанализируй запрос и собери look, который бы удовлетворял всем требованиям пользователя.
Наряд базово может состоять из top+bottom или full.
Также выбери пол, сезон, укажи обувь. При необходимости добавь верхнюю одежду/аксессуары.
Одно значение может состоять только из одного слова. 
Для обозначения sex используй female, male, unisex. Для остального только русский язык. 
"""
TOTAL_CREATIONLOOK_PROMPT = """\
You are a professional stylist creating a total look.
Here's the user request: {request}.
Analyze the request and put together a look that meets all the user's requirements.
The basic outfit can consist of top+bottom or full.
Also select gender, season, specify shoes. Add outerwear/accessories if needed.
One value can consist of only one word.
For sex use female, male, unisex. For the rest, use Russian only.
"""

# ---------- Pydantic models ----------
class Item(BaseModel):
    category: str = Field(..., description="название категории одежды одним словом")
    fabric:  Optional[str] = None
    color:   Optional[str] = Field(None, description="Use simple one-word color names (розовый, черный, и т.п.); avoid complex shades. If necessary, you may use светлый or темный instead of a specific color.")  
    pattern: Optional[str] = None
    fit:     Optional[str]       = None


class OneTotalLook(BaseModel):
    sex:        str  = Field(..., description="female, male, unisex")
    season:     Optional[str]      = Field(None, description="зимний, летний и т.п.")
    top:        Optional[List[Item]] = None
    bottom:     Optional[List[Item]] = None
    full:       Optional[List[Item]] = None
    shoes:      List[Item]
    outerwear:  Optional[List[Item]] = None
    accessories: Optional[List[Item]] = None