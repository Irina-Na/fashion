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
    fit: str            # fitted | semi‑fitted | oversize
    sleeve_len: str = Field(..., description="long, 3/4, short, non-sleeve")
    neckline: List[str] = Field(..., description="shape and depth")
    collar: Optional[str] = Field(..., description="shape and height")
    closure: Optional[str]
    pockets: str
    top_length: str = Field(..., description= "crop | reg | long | tunic | ext")
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    fabric: List[str]
    surfase: str
    textured_surface_type: Optional[str]
    cut_features: List[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str] 
    confidence: float = Field(..., ge=0, le=1)

class BottomItem(BaseModel):
    """Attributes for lower‑body garments (trousers, skirts, shorts)."""
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    waistline: str = Field(..., description="high, mid, low")
    waistband: str
    pockets: str
    length: str
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    fabric: List[str]
    surfase: str
    textured_surface_type: Optional[str]
    cut_features: List[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str] 
    confidence: float = Field(..., ge=0, le=1)


class FullBodyItem(BaseModel):
    """Attributes for dresses, jumpsuits, overalls — single full‑body pieces."""
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    sleeve_len: str = Field(..., description="long, 3/4, short, non-sleeve")
    neckline: List[str] = Field(..., description="shape and depth")
    collar: Optional[str] = Field(..., description="shape and height")
    waistline: str = Field(..., description="high, mid, low, no-defined")
    waistband: str
    pockets: str 
    length: str         # mini | midi | maxi 
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    fabric: List[str]
    surfase: str
    textured_surface_type: Optional[str]
    cut_features: List[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str]
    confidence: float = Field(..., ge=0, le=1)

class OuterwearItem(BaseModel):
    """Attributes for coats, jackets, parkas — garments worn over main outfit."""
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    sleeve_len: str = Field(..., description="long, 3/4, short, non-sleeve")
    collar: str = Field(..., description="shape and height")
    waistline: Optional[str] = Field(..., description="high, mid, low, no-defined")
    closure: Optional[str] 
    pockets: str 
    length: str         # mini | midi | maxi 
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    fabric: List[str]
    surfase: str
    textured_surface_type: Optional[str]
    cut_features: List[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str]
    confidence: float = Field(..., ge=0, le=1)


class ShoesItem(BaseModel):
    """Attributes for footwear."""
    category: str
    sole_profile: str  = Field(..., description="flat, heel, tankette, platform, high heel")
    shank_height: str  = Field(..., description="hight, middle, low")
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    fabric: List[str]
    surfase: str
    textured_surface_type: Optional[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str]
    confidence: float = Field(..., ge=0, le=1)


class BagItem(BaseModel):
    """Attributes for bags, backpacks, clutches."""
    category: str
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    fabric: List[str]
    surfase: str
    textured_surface_type: Optional[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str]
    confidence: float = Field(..., ge=0, le=1)


class AccessoryItem(BaseModel):
    """Attributes for miscellaneous accessories: belts, hats, jewelry, scarves, etc."""
    category: str
    model_construction: List[str]
    sex: str            # f | m | u
    pattern: str
    color_temperature: str
    color_tone: str
    color_hsl: list[ColorHSL]
    material: List[str]
    season: str         # summer | demi | winter
    base: bool
    style: List[str]
    confidence: float = Field(..., ge=0, le=1)

# ────────────────────────────────────────────────────────────────────────────────
# Templates: per‑meta‑category configuration for LLM calls
# ────────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "top": {
        "schema": TopItem.model_json_schema(),
        "class": TopItem,
        "model": "gpt-4.1-mini",
        "fewshots_categories": "(e.g. polo, shirt, tee, top, tank top, blazer, etc.)",
        "fewshots_silhouette": "(e.g. for top: crop top, halter top, tube top; for tank top - None)",
    }, 
    "bottom": {
        "schema": BottomItem.model_json_schema(),
        "class": BottomItem,
        "model": "gpt-4.1-mini", 
        "fewshots_categories": "(e.g. pants, skirt, jeans, etc.)",   
        "fewshots_silhouette":"(e.g. for pants and jeans: skinny, palazzo, pencil, straight, etc.)",
    },
    "fullbody": {
        "schema": FullBodyItem.model_json_schema(),
        "class": FullBodyItem,
        "model": "gpt-4.1-mini",
        "fewshots_categories": "(e.g. dress, suit, set, etc.)",
        "fewshots_silhouette": "(e.g. for dress: straight, cocoon, wrap, etc.)",
    },
    "outwear": {
        "schema": OuterwearItem.model_json_schema(),
        "class": OuterwearItem,
        "model": "gpt-4.1-mini",
        "fewshots_categories": "(e.g. coat, parka, puffer, cape, etc.)",
        "fewshots_silhouette":"(e.g. for coat: straight, cocoon, wrap, etc.)",
    },
    "shoes": {
        "schema": ShoesItem.model_json_schema(),
        "class": ShoesItem,
        "model": "gpt-4.1-nano",
        "fewshots_categories": "loafers, sneakers, boots, booties, pumps, flats, sandals",
        "fewshots_silhouette": "(e.g. for sneakers: sneaker, running, dad-shoes, etc.)",
    },
    "bag": {
        "schema": BagItem.model_json_schema(),
        "class": BagItem,
        "model": "gpt-4.1-nano",
        "fewshots_categories": "(one of: clutch, backpack, crossbody, belt bag, tote, shopper, briefcase)", 
        "fewshots_silhouette": '',
    },
    "accessorize": {
        "schema": AccessoryItem.model_json_schema(),
        "class": AccessoryItem,
        "model": "gpt-4.1-nano",
        "fewshots_categories": "(e.g. watch, scarf, hat, shawl, tie, bracelet, etc.)",
        "fewshots_silhouette":'',
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
Where top is upper-body garments designed to be worn as the primary visible layer — directly on skin or over a base piece (shirt, blouse, vests, sweater) — excluding havy outerwear, that worn over main outfit (coat, down jacket, etc.).
Fullbody - dress, suit, jumpsuit, set, etc.
2. Provide confidence level from 0.0 to 1.0 based on how certain you are about the classification.
'''


# NOTE: Replace placeholders {…} with actual strings or variables containing the relevant
# enum lists and few‑shot examples before using `TEMPLATES` in production code.
#(female, male, unisex)

GENERAL_PROMPT  = f'''
You are a fashion-attribute extractor.

### Global rules.  
• `category` → top-level product type label used for routing taxonomy and **META_CATEGORY** vocabularies **CATEGORY_EXAMPLES**.
• `sex` → `f` | `m` | `u`.  Mean: female, male, unisex.
• `fit` → `fitted` | `semi-fitted` | `oversized`.
• `length` → `mini`, `midi`, `maxi`.
• `pockets` → one of: `non` or type of poket - e.g. kangaroo, faux, cargo, etc.
• `pattern` → `no-print` | `colorblock` | `abstract` | `animal` | `watercolor` | `checked` | `striped-horizontal` | `striped-vertical` | `geometric` | `lettering-emblem` | `military` | `polka-dot` | `ethno` | `floral` | `crushed` | `draped` | `pleated`. Visible lines also count as pattern.
• `color_temperature` →  `warm`| `cold` | `achromatic`. Most colors can be warm or cool, depending on their yellow or blue undertones.
• `color_tone` → `pastel` | `bright` | `muted` | `dark-shades`. Pastel=base+white (light), muted=base+little black, dark-shades=base+much black, bright=pure base color.
• `color_hsl` must be an array `[H, S, L]` of three integers:
  – `H`  0-360,  `S` 0-100,  `L` 0-100.  If colorblock pattern, of few colors, list them.
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
• `surface` - one or a list of the following options:
**matte** – flat, non-reflective surface.  
**semi-matte** – flat surface, slightly reflective, soft sheen.  
**shiny** – flat surface, glossy, light-reflecting finish. 
**sheer/transparent** - gauzy, airy, see-through surface (e.g., chiffon, organza).
**textured** - garment have other the Visible or tactile relief with uneven structure.
• `textured_surface_type` - if `surface = textured`, specify the exact texture (e.g., tweed, boucle, ribbed, crinkled, drapping).
• `season` → One of: `summer` (only summer wear), `demi` (multi-season), `winter` (only winter wear).
• `model_construction` → category-specific canonical cut/shape/silhouette label if exist **MODEL_EXAMPLES** or сonstruction features not mentioned above.
• `cut_features`→ multi-tag field for intentional **patternmaking** and **construction** techniques not mentioned above. Records stable design choices—**shaping**, **openings**, **paneling & seam placement**, **line direction/symmetry**, **edge finishes**, **fastening setup**, **internal supports**—сonstructions features that remain legible in wear and motion.
• `base` - boolean, is the garment basic? A basic garment is an element of a casual wardrobe with: a simple, straight, clean cut (no ruffles, drapery, complex construction, complex asymmetry, or designer “tricks”);
garments that qualify as basic = those with a straightforward cut. Surface: the shape is simple, but the surface can be interesting (fabric, texture, prints or color), but without complicating decorative details (construction-intensive; many pieces and style lines; advanced shaping/volume (draping, pleating, layering, engineered openings); complex finishes and internal structure.).
• `style` - one or a list of the following options. Assign every style whose criteria it meets. A single garment can carry multiple labels:
**safari**  
Natural fabrics (cotton, linen), sandy, khaki, olive tones. Pockets, belts, lacing, metal fittings. Functional with expedition character.
**military**  
Strict silhouettes, uniform-like cuts, protective fabrics, khaki tones, camouflage, epaulettes, brass buttons. Military uniform attributes in civil fashion.
**marine**  
Nautical theme: stripes, white-blue-red palette, sailor collars, telnyashka, golden buttons. Light fabrics for summer leisure, accent on freshness and “yachting chic”.
**drama**  
Theatrical effect, sexuality, aggression and luxury. Leather, latex, sequins, deep cuts, asymmetry, drapery. Black and jewel tones. “Wow-factor” items.
**romantic**  
Soft lines, light fabrics (silk, chiffon, lace), pastel tones, floral prints. Ruffles, bows, flounces. Silhouettes emphasize tenderness and refinement.
**feminine**  
Emphasizes figure (fitted silhouettes, skirts, dresses), soft fabrics, elegant accessories. Palette from pastels to saturated, but always graceful.
**jockey**  
Equestrian-inspired: riding boots, breeches, slim trousers, vests, redingote jackets, leather gloves, jockey caps. Colors — cognac, black, burgundy.
**dandy**  
Men’s wardrobe with refined irony: perfectly tailored suits, waistcoats, ties, canes, hats. Luxurious fabrics, sometimes with vintage touch. Palette ranges from classic to bold accents.
**retro**  
Clothing referencing past decades (20s, 50s, 70s). Characteristic silhouettes, prints, fabrics: pleats, flares, polka dots, vinyl, tweed.
**ethnic**  
National motifs: embroidery, ornaments, ethnic prints, folk fabrics (wool, linen, cotton). Loose silhouettes, amulet-like accessories.
**boho**  
Freedom and layering: long skirts, loose dresses, vests, fringe, ethnic jewelry. Natural fabrics, warm earthy tones, mixed patterns. More bohemian than ethnic.
**minimalism**  
Avant-garde style. Maximum simplicity and “silence”: clear geometric silhouettes, monochrome, no decor. Focus on form, proportions and fabric texture.
**de-constructivism**  
Avant-garde style. Deliberate asymmetry, “broken” lines, inside-out seams, torn/reassembled elements. Experimental reinterpretation of clothing.
**conceptualism**  
Avant-garde style. Clothing as idea or manifesto. May be sculptural or theatrical, not necessarily wearable. Unusual materials and forms; concept more important than practicality.
**classic**  
Casual style, but strict protocol clothing for officials: suit sets, closed dresses, calm palette (navy, grey, black, beige), minimal jewelry. Purpose — reliability and restraint, not fashion. timeless “reliable uniform”. Very rare. Qween family, etc.
**business-best**  
Casual style, but Modern version of classic for conservative professions (diplomats, bankers, lawyers). Strict suits, restrained colors (navy, grey, burgundy, emerald), small checks or stripes allowed. Contemporary cuts without extremes. Goal — status and trust.
**business-casual**  
Casual style, but Less strict business style: blazers, trousers, pencil skirts, blouses, soft suit fabrics. Neutral with muted accents. Allows modern silhouettes, textures and accessories. Professional but not rigid.
Common in conservative industries focused on money and reputation: finance, law, pharmaceuticals. Usually explicitly prescribed and followed at all levels, especially by employees with representative functions — the “face of the company.” Modern cuts combined with a system of restrictions: clothing should minimize fuss, inspire trust, and convey stability and reliability — not flashy success, but steady professionalism.
**smart-casual**  
Casual style, but Appropriate for both conservative and creative professions without a strict dress code. Balances fashion, chic, and individuality, allowing more freedom of self-expression while keeping a polished look.
Relaxed business style with casual elements: blazer + jeans, shirt + chinos, simple dresses. Items combine comfort and respectability. Wider palette than business-casual but without excess.
**city-casual**  
Everyday basic casual clothes: jeans, t-shirts, sweatshirts, sneakers. Focus on comfort, practicality, and simplicity.
• `confidence` is a float **0–1** (0.75 = medium-sure).

### Instructions
1. Use **only** the exact values listed above.
2. Every key must be fill - choose the closest value. Never leave any value unfilled.  
3. Language of input may be Russian or English; output enums are **always English**. 
4. Analyze the image of the following item description: **NAME**
5. Trust the image more then description.
'''


'''
Колорблок!!
• `cut_features` - any 
slits, cut-outs, neckline, off-shoulder, raglan, batwing, puffed or bishop sleeves, ruffled, raw-edge, wrap, peplum, empire, a-siluet, etc.


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