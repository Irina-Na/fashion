"""Pydantic schemas and LLM call templates for fashion attribute extraction.
The file contains:
  • Seven minimal BaseModel subclasses – one for each meta‑category.
  • A global `TEMPLATES` dict: maps meta‑category → {schema, model, enum_blocks, fewshots}.
Static strings like `TOP_ENUMS_TXT` or `SHOES_FEWSHOTS` must be defined in calling code (loaded from .txt files or literals).
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, TypeAlias
from pydantic import BaseModel, Field, conlist, conint

# ────────────────────────────────────────────────────────────────────────────────
# Common constrained types
# ────────────────────────────────────────────────────────────────────────────────
#ColorHSL: TypeAlias = conlist(conint(ge=0, le=360), min_items=3, max_items=3)  # type alias!

#ColorHSL = conlist(conint(ge=0, le=360), min_items=3, max_items=3)  # type alias!
ColorHSL = conlist(conint(ge=0, le=360), min_length=3, max_length=3) 
# ────────────────────────────────────────────────────────────────────────────────
# Pydantic models (one per meta‑category)
# ────────────────────────────────────────────────────────────────────────────────

class TopItem(BaseModel):
    """Attributes for upper‑body garments (shirts, blouses, sweaters, etc.)."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    #length: str = None         # cropped | waist | hip
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    сonstruction: List[str] = Field(default_factory=list) 
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)


class BottomItem(BaseModel):
    """Attributes for lower‑body garments (trousers, skirts, shorts)."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    сonstruction: List[str] = Field(default_factory=list)
    length: str = None         # mini | midi | maxi 
    waist_fit: str = None          # high, standart, low
    garment_type: str  = Field(..., description="e.g. skinny, palazzo, pencil, A‑line")
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)


class FullBodyItem(BaseModel):
    """Attributes for dresses, jumpsuits, overalls — single full‑body pieces."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    сonstruction: List[str] = Field(default_factory=list)
    length: str = None         # mini | midi | maxi 
    waist_fit: str = None          # high, standart, low
    garment_type: str  = Field(..., description="e.g. A‑silhouette")
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)

class OuterwearItem(BaseModel):
    """Attributes for coats, jackets, parkas — garments worn over main outfit."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    сonstruction: List[str] = Field(default_factory=list)
    length: str = None         # mini | midi | maxi 
    garment_type: str  = Field(..., description=" e.g. A‑silhouette")
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)


class ShoesItem(BaseModel):
    """Attributes for footwear."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    garment_type: str  = Field(..., description="flat, midle heel, high heel, platform")
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)



class BagItem(BaseModel):
    """Attributes for bags, backpacks, clutches."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)


class AccessoryItem(BaseModel):
    """Attributes for miscellaneous accessories: belts, hats, jewelry, scarves, etc."""
    category: str
    sex: str = None            # f | m | u
    season: str = None         # summer | demi | winter
    fit: str = None            # fitted | semi‑fitted | oversize
    fabric: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    color_hsl: ColorHSL = None
    color_temperature: str = None
    color_tone: str = None
    pattern: List[str] = Field(default_factory=list)
    style: List[str] = Field(default_factory=list) 
    confidence: float = Field(..., ge=0, le=1)

# ────────────────────────────────────────────────────────────────────────────────
# Templates: per‑meta‑category configuration for LLM calls
# ────────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "top": {
        "schema": TopItem.model_json_schema(),
        "class": TopItem,
        "model": "gpt-4.1-mini",
        "enum_blocks": "\n".join([  
             "• `length` → cropped, waist, hip",
            "• `fit` → fitted, semi-fitted, oversize.",
        ]),
        "fewshots": "{TOP_FEWSHOTS}",
    },
    "bottom": {
        "schema": BottomItem.model_json_schema(),
        "class": BottomItem,
        "model": "gpt-4.1-mini",
        "enum_blocks": "\n".join([
            "• `garment_type`  → ", "{BOTTOM_GARMENT_TYPES_TXT}",
            "• `length` → mini, midi, maxi",
            "• `fit` → fitted, semi-fitted, oversize.",
            "• `waist_fit` → high, standart, low.",
        ]),
        "fewshots": "{BOTTOM_FEWSHOTS}",
    },
    "fullbody": {
        "schema": FullBodyItem.model_json_schema(),
        "class": FullBodyItem,
        "model": "gpt-4.1-mini",
        "enum_blocks": "\n".join([
            "• `garment_type`  → ", "{FULLBODY_GARMENT_TYPES_TXT}", 
            "• `length`  → mini, midi, maxi",
            "• `fit`  → fitted, semi-fitted, oversize.",
            "• `waist_fit`  →  high, standart, low.",

        ]),
        "fewshots": "{FULLBODY_FEWSHOTS}",
    },
    "outwear": {
        "schema": OuterwearItem.model_json_schema(),
        "class": OuterwearItem,
        "model": "gpt-4.1-mini",
        "enum_blocks": "\n".join([
            "• `garment_type` → ", "{OUTERWEAR_GARMENT_TYPES_TXT}", 
            "• `length` → mini, midi, maxi",
            "• `fit` → fitted, semi-fitted, oversize.",
        ]),
        "fewshots": "{OUTERWEAR_FEWSHOTS}",
    },
    "shoes": {
        "schema": ShoesItem.model_json_schema(),
        "class": ShoesItem,
        "model": "gpt-4.1-nano",
        "enum_blocks": "\n".join([
            "• `garment_type`  →  flat, midle heel, high heel, platform"
        ]),
        "fewshots": "{SHOES_FEWSHOTS}",
    },
    "bag": {
        "schema": BagItem.model_json_schema(),
        "class": BagItem,
        "model": "gpt-4.1-nano",
        "enum_blocks":"",
        "fewshots": "{BAG_FEWSHOTS}",
    },
    "accessorize": {
        "schema": AccessoryItem.model_json_schema(),
        "class": AccessoryItem,
        "model": "gpt-4.1-nano",
        "enum_blocks": [],
        "fewshots": "{ACC_FEWSHOTS}",
    },
}

class CategoryType(BaseModel):   
    category: str

class MetaCategory(BaseModel):
    """Schema for meta-category classification."""
    category: str
    confidence: float = Field(..., ge=0, le=1, description="Confidence level for the classification")

META_CATEGORY_DETECTION_PROMPT  = f'''
You are a fashion-attribute extractor.  
### Instructions:
1. Classify the description of the item into one of the categories below:  'top', 'bottom', 'fullbody', 'outerwear', 'shoes', 'bag', 'accessories'.
Where top is upper-body garments designed to be worn as the primary visible layer — directly on skin or over a base piece (shirt, blouse, vests, sweater) — excluding havy outerwear (coat, down jacket, etc.).
2. Provide confidence level from 0.0 to 1.0 based on how certain you are about the classification
'''


# NOTE: Replace placeholders {…} with actual strings or variables containing the relevant
# enum lists and few‑shot examples before using `TEMPLATES` in production code.

GENERAL_PROMPT  = '''
You are a fashion-attribute extractor.

### Global rules. 
• `sex` → `f` | `m` | `u`  (female, male, unisex).  
• `season`  → `summer` | `demi` | `winter`.
• `fit` → `fitted` | `semi-fitted` | `oversized`.
• `waist_fit` → `high`, `standart`, `low`.
• `length` → `mini`, `midi`, `maxi`.
• `color_hsl` must be an array `[H, S, L]` of three integers:
  – `H`  0-360,  `S` 0-100,  `L` 0-100.  
• `color_temperature` →  `warm`| `cold` | `achromatic`. Most colors can be warm or cool, depending on their yellow or blue undertones.
• `color_tone` → `pastel` | `bright` | `muted` | `dark-shades` | `neutral-palette`.
• `patterns` → `no-print` | `abstract` | `animal` | `watercolor` | `checked` | `ethno` | `floral` | `geometric` | `lettering-emblem` | `military` | `polka-dot` | `crushed` | `draped` | `pleated`
(despite what it says about the shape, we will categorize it as a pattern, because having visible lines on the garment is also a pattern that should be taken into account to not overwhelm the look or make it interesting.
• `fabric` - use the most appropriate or fill in with your own:
  - angora
  - boucle & tweed
  - cashmere
  - chiffon
  - corduroy
  - cotton
  - crepe
  - cutout lace & eyelash
  - denim
  - fur
  - jacquard
  - knitwear & mohair
  - leather
  - linen
  - organza
  - suede
  - taffeta
  - velvet
  - wool
  - knitwear
  - mohair
  - fleece
  - boucle
  - nylon
  - silk
  - tweed
  - elasticized
  - gabardine
  - satin.
• `сonstruction` one or a list of the following options: `simple` | `minimalistic` | `complex` | `pleats` | `draping` | `cut-outs` | `slits`.
• `material`  one or a list of the following options: `matte` | `semi-matte` | `shiny` | `rigid` | `structured` | `cozy` | `draping` | `thin` | `voluminous` | `textured` | `neutral-texture` | `unusual` | `high-tech`.
• `confidence` is a float **0–1** (0.75 = medium-sure).
• `style` one or a list of the following options: classic, bussiness-best, bussiness-casual, smart-casual, casual(base),  safari, military, marine, drama, romantic, feminine, jockey, dandy, retro, entic (boho), avant-garde.
Instruction: For each garment description or image, assign every style whose criteria it meets. A match requires at least three of the listed criteria (silhouette, materials, colours/prints, unique markers). A single garment can carry multiple labels.

**Safari**  
Natural fabrics (cotton, linen), sandy, khaki, olive tones. Pockets, belts, lacing, metal fittings. Functional with expedition character.

**Military**  
Strict silhouettes, uniform-like cuts, protective fabrics, khaki tones, camouflage, epaulettes, brass buttons. Military uniform attributes in civil fashion.

**Marine**  
Nautical theme: stripes, white-blue-red palette, sailor collars, telnyashka, golden buttons. Light fabrics for summer leisure, accent on freshness and “yachting chic”.

**Drama**  
Theatrical effect, sexuality, aggression and luxury. Leather, latex, sequins, deep cuts, asymmetry, drapery. Black and jewel tones. “Wow-factor” items.

**Romantic**  
Soft lines, light fabrics (silk, chiffon, lace), pastel tones, floral prints. Ruffles, bows, flounces. Silhouettes emphasize tenderness and refinement.

**Feminine**  
Emphasizes figure (fitted silhouettes, skirts, dresses), soft fabrics, elegant accessories. Palette from pastels to saturated, but always graceful.

**Jockey**  
Equestrian-inspired: riding boots, breeches, slim trousers, vests, redingote jackets, leather gloves, jockey caps. Colors — cognac, black, burgundy.

**Dandy**  
Men’s wardrobe with refined irony: perfectly tailored suits, waistcoats, ties, canes, hats. Luxurious fabrics, sometimes with vintage touch. Palette ranges from classic to bold accents.

**Retro**  
Clothing referencing past decades (20s, 50s, 70s). Characteristic silhouettes, prints, fabrics: pleats, flares, polka dots, vinyl, tweed.

**Ethnic**  
National motifs: embroidery, ornaments, ethnic prints, folk fabrics (wool, linen, cotton). Loose silhouettes, amulet-like accessories.

**Boho**  
Freedom and layering: long skirts, loose dresses, vests, fringe, ethnic jewelry. Natural fabrics, warm earthy tones, mixed patterns. More bohemian than ethnic.

**Avant-garde (minimalism)**  
Maximum simplicity and “silence”: clear geometric silhouettes, monochrome, no decor. Focus on form, proportions and fabric texture.

**Avant-garde (de-constructivism)**  
Deliberate asymmetry, “broken” lines, inside-out seams, torn/reassembled elements. Experimental reinterpretation of clothing.

**Avant-garde (conceptualism)**  
Clothing as idea or manifesto. May be sculptural or theatrical, not necessarily wearable. Unusual materials and forms; concept more important than practicality.

**Classic**  
Strict protocol clothing for officials: suit sets, closed dresses, calm palette (navy, grey, black, beige), minimal jewelry. Purpose — reliability and restraint, not fashion. timeless “reliable uniform”. Very rare. Qween family, etc.

**Business-best**  
Modern version of classic for conservative professions (diplomats, bankers, lawyers). Strict suits, restrained colors (navy, grey, burgundy, emerald), small checks or stripes allowed. Contemporary cuts without extremes. Goal — status and trust.

**Business-casual**  
Less strict business style: blazers, trousers, pencil skirts, blouses, soft suit fabrics. Neutral with muted accents. Allows modern silhouettes, textures and accessories. Professional but not rigid.
Common in conservative industries focused on money and reputation: finance, law, pharmaceuticals. Usually explicitly prescribed and followed at all levels, especially by employees with representative functions — the “face of the company.” Modern cuts combined with a system of restrictions: clothing should minimize fuss, inspire trust, and convey stability and reliability — not flashy success, but steady professionalism.

**Smart-casual**  Appropriate for both conservative and creative professions without a strict dress code. Balances fashion, chic, and individuality, allowing more freedom of self-expression while keeping a polished look.
Relaxed business style with casual elements: blazer + jeans, shirt + chinos, simple dresses. Items combine comfort and respectability. Wider palette than business-casual but without excess.

**Casual (base)**  
Everyday basic style. City casual = BASE — Basic items that do not belong to any specific style.
A basic garment is an element of a casual wardrobe with: a simple, straight, clean cut (no ruffles, drapery, complex asymmetry, or designer “tricks”);
garments that qualify as basic = those with a straightforward cut.
Surface: the shape is simple, but the surface can be interesting (fabric, texture, or color) so it doesn’t look boring. The surface may include texture or prints, but without complicating decorative details.
"""

### Instructions
1. Use **only** the exact enum values listed above, exclude fabric.
2. Never invent new keys; every key must exist in the provided schema.  
3. Language of input may be Russian or English; output enums are **always English**. 
4. If a value cannot be inferred with reasonable certainty, output:
  – `null`   for scalars,  
  – `[]`     for lists. 
'''


'''
Safari
Colonial-inspired style: shirt-dresses, jackets with patch pockets, belts, natural fabrics (cotton, linen), earthy palette (khaki, beige, sand). Functional and natural.

Military
Military aesthetic: camouflage, epaulettes, brass buttons, coats, parkas. Color scheme: olive, khaki, navy, grey. Stricter and more utilitarian than Safari.

Marine
Nautical style: striped shirts, pea coats, double-breasted blazers, white-blue-red palette, gold buttons. Light and maritime, unlike the harsher Military.

Drama
Theatrical, statement-making style. Armor-like silhouettes (broad shoulders, cinched waist), asymmetry, draping, leather, vinyl, sequins, metallic fabrics, spikes, chains. Dominant black with bright accents. Aimed at provocation, luxury, and sexuality.

Romantic
Soft and tender style. Light fabrics (silk, chiffon), pastel palette, ruffles, bows, floral prints. Creates airy and dreamy femininity. Unlike Drama, it is gentle rather than aggressive.

Feminine
Womanly style emphasizing body shape. Dresses, skirts, heels, waist-accented cuts, soft lines. Can be casual or festive. Less “fairy-like” than Romantic, more focused on sensuality.

Jockey
Equestrian style. Riding boots, slim pants (jodhpurs), tailored jackets, leather belts, gloves. Typical colors: black, brown, white, burgundy. Defined by horse-riding aesthetics.

Dandy
Masculine-inspired chic in women’s fashion: tailored suits, waistcoats, ties, bowler hats. Neutral palette, suiting fabrics. Stresses polished, masculine elegance.

Retro
Looks inspired by past decades (20s–80s). Examples: flared skirts, 60s mini dresses, 80s shoulder-padded blazers. Defined by historical reference.

Ethnic
National-inspired style: embroidery, tribal or folk ornaments, traditional fabrics. Strong link to a specific cultural heritage (African, Asian, Slavic, etc.).

Boho
Bohemian eclectic style: loose silhouettes, layering, natural fabrics, fringe, lace, oversized shapes, mixed prints. Relaxed and artistic.

Avant-garde (minimalism)
Pure forms and reduced decoration. Monochrome, strict lines, restrained shapes. Focus on form itself. Cold intellectual minimalism.

Avant-garde (de-constructivism)
Deconstructed fashion: asymmetry, raw edges, unfinished seams, unconventional garment structure. Garments appear “taken apart and reassembled.” Breaks form deliberately.

Avant-garde (conceptualism)
Fashion as artistic statement. Uses unusual materials, symbols, or text. Often impractical, prioritizing concept over wearability.

Classic
Formal representative style tied to diplomatic and protocol dress codes. Features strict tailoring, covered silhouettes, neutral palette (navy, grey, black, beige), minimal jewelry (pearls, discreet watches). Conveys reliability, modesty, and authority.

Business-best
Modern conservative business style (successor of Classic). Used in finance, law, banking, government. Based on suits (trousers or skirt with jacket), muted or monochrome colors, discreet prints (pinstripe, check, houndstooth). Must look modern in cut and fabric.

Business-casual
Relaxed business style: combines formal elements (blazer, trousers) with softer ones (cardigan, blouse, flat shoes). Allows individuality while maintaining professionalism. Broader color and cut options than business-best.

Smart-casual
“Smart everyday” style balancing comfort and neatness. Includes blazers, dark jeans, minimalist dresses, neat shirts. Works both for offices without dress code and daily wear.

Casual (base)
Basic everyday style: jeans, t-shirts, sweatshirts, sneakers, simple jackets. Focus on comfort, practicality, and simplicity.

'''