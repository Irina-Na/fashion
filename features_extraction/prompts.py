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

class UpperBodyItem(BaseModel):
    """Attributes for upper‑body garments (shirts, blouses, sweaters, etc.)."""
    consistency_check: str
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    sleeve_len: str = Field(..., description="long, 3/4, short, non-sleeve")
    sleeve: List[str] = Field(..., description="shape and depth")
    neckline: List[str] = Field(..., description="shape and depth")
    collar: Optional[str] = Field(..., description="shape and height")
    closure: Optional[str]
    pockets: str
    top_length: str = Field(..., description= "`crop`- ribcage level, `reg` - waistline, `long` - hip, `tunic` - cover hip, `ext` - longer. If assimetric, specify both lengths.")
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

class LowerBodyItem(BaseModel):
    """Attributes for lower‑body garments (trousers, skirts, shorts)."""
    consistency_check: str
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    waistline: str = Field(..., description="high, mid, low")
    waistband: str
    inseam: str = Field(..., description="short, mid, long. If assimetric, specify both lengths.")
    outseam: str = Field(..., description="short, mid, long. If assimetric, specify both lengths.")
    crotch_height: str 
    leg_opening: str 
    leg_shape: str
    closure: Optional[str]
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
    consistency_check: str
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    sleeve_len: str = Field(..., description="long, 3/4, short, non-sleeve")
    sleeve: List[str] = Field(..., description="shape and depth")
    neckline: List[str] = Field(..., description="shape and depth")
    collar: Optional[str] = Field(..., description="shape and height")
    waistline: str = Field(..., description="high, mid, low, no-defined")
    waistband: str
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

class OuterwearItem(BaseModel):
    """Attributes for coats, jackets, parkas — garments worn over main outfit."""
    consistency_check: str
    category: str
    fit: str            # fitted | semi‑fitted | oversize
    sleeve_len: str = Field(..., description="long, 3/4, short, non-sleeve")
    sleeve: List[str] = Field(..., description="shape and depth")
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
    consistency_check: str
    category: str
    sole_profile: str  = Field(..., description="flat, heel, tankette, platform, high heel")
    shank_height: str  = Field(..., description="hight, middle, low")
    closure: str
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
    consistency_check: str
    category: str
    closure: str
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
    """Attributes for accessories: belts, hats, jewelry, scarves, etc."""
    consistency_check: str
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

# NOTE: Replace placeholders {…} with actual strings or variables containing the relevant
# enum lists and few‑shot examples before using `TEMPLATES` in production code.
#(female, male, unisex)


TEMPLATES: Dict[str, Dict[str, Any]] = {
    "top": {
        "schema": UpperBodyItem.model_json_schema(),
        "class": UpperBodyItem,
        "model": "gpt-4.1-mini",
        "metacategory_name": "Upper‑body garments",
        "fewshots_categories": "(e.g. polo, shirt, tee, top, tank top, blazer, etc.)",
        "fewshots_silhouette": "(e.g. for top: crop top, halter top, tube top; for tank top - None)",
    }, 
    "bottom": {
        "schema": LowerBodyItem.model_json_schema(),
        "class": LowerBodyItem,
        "model": "gpt-4.1-mini", 
        "metacategory_name": "Lower‑body garments",
        "fewshots_categories": "(e.g. pants, skirt, jeans, etc.)",   
        "fewshots_silhouette":"(e.g. for pants and jeans: skinny, palazzo, pencil, straight, etc.)",
    },
    "fullbody": {
        "schema": FullBodyItem.model_json_schema(),
        "class": FullBodyItem,
        "model": "gpt-4.1-mini",
        "metacategory_name": "FullBody",
        "fewshots_categories": "(e.g. dress, suit, set, etc.)",
        "fewshots_silhouette": "(e.g. for dress: straight, cocoon, wrap, etc.)",
    },
    "outwear": {
        "schema": OuterwearItem.model_json_schema(),
        "class": OuterwearItem,
        "model": "gpt-4.1-mini",
        "metacategory_name": "Outerwear",
        "fewshots_categories": "(e.g. coat, parka, puffer, cape, etc.)",
        "fewshots_silhouette":"(e.g. for coat: straight, cocoon, wrap, etc.)",
    },
    "shoes": {
        "schema": ShoesItem.model_json_schema(),
        "class": ShoesItem,
        "model": "gpt-4.1-nano",
        "metacategory_name": "Footwear",
        "fewshots_categories": "loafers, sneakers, boots, booties, pumps, ballet, slip-ons, etc.",
        "fewshots_silhouette": "(e.g. for sneakers: sneaker, running, dad-shoes, etc., for boots: chelsea, combat, biker, etc.)",
    },
    "bag": {
        "schema": BagItem.model_json_schema(),
        "class": BagItem,
        "model": "gpt-4.1-nano",
        "metacategory_name": "Bags",
        "fewshots_categories": "(one of: clutch, backpack, crossbody, belt bag, tote, shopper, briefcase)", 
        "fewshots_silhouette": '',
    },
    "accessorize": {
        "schema": AccessoryItem.model_json_schema(),
        "class": AccessoryItem,
        "model": "gpt-4.1-nano",
        "metacategory_name": "Accessories",
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

GENERAL_PROMPT  = f'''
You are a fashion-attribute extractor.

### Global rules.  
• `sex` → `f` | `m` | `u`.  Mean: female, male, unisex.
• `fit` → `fitted` | `semi-fitted` | `oversized`.
• `length` → `mini`, `midi` (from knee to lower calf), `maxi`(from lower calf to the foot). If assimetric, specify both lengths.
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
**matte** - flat, non-reflective surface.  
**semi-matte** - flat surface, slightly reflective, soft sheen.  
**shiny** - flat surface, glossy, light-reflecting finish. 
**sheer/transparent** - gauzy, airy, see-through surface (e.g., chiffon, organza).
**textured** - garment have other the Visible or tactile relief with uneven structure.
• `textured_surface_type` - if `surface = textured`, specify the exact texture (e.g., tweed, boucle, ribbed, crinkled, drapping).
• `season` → One of: `summer` (only summer wear), `demi` (multi-season), `winter` (only winter wear).
• `base` - boolean, is the garment basic? A basic garment is an element of a casual wardrobe with: a simple, straight, clean cut (no ruffles, drapery, complex construction, complex asymmetry, or designer “tricks”);
garments that qualify as basic = those with a straightforward cut. Surface: the shape is simple, but the surface can be interesting (fabric, texture, prints or color), but without complicating decorative details (construction-intensive; many pieces and style lines; advanced shaping/volume (draping, pleating, layering, engineered openings); complex finishes and internal structure.).
• `style` - one or more of the following styles. A single garment may fit multiple labels. Usualy it's a base + another style. Rarely two non-base styles. Use the descriptions below to understand each style’s overall vibe rather than focusing on strict traits. If an item matches the vibe of a style, assign that style.
**military**  
Uniform roots after WWI, reinforced post-WWII, revived 1970s protest, embedded in 1980s–1990s rock and grunge, remains popular. Structured silhouette, dense fabrics, epaulettes, double-breasted plackets, patch pockets, straps, lacing. Palette: khaki, pistachio, swamp green, navy, grey, black. Details: zippers, matte metal buttons, badges, insignia. Footwear: combat boots, rough utilitarian shoes. Impression: strict, disciplined, utilitarian. Quick ID: khaki field jacket with epaulettes. Contrast with Marine: breezy light forms vs. strict structure.
**safari**  
Colonial-inspired style: British uniforms for Africa, transformed by Yves Saint Laurent “Saharienne” dress 1968. Cut close to military: epaulettes, pockets, plackets, but in sand and earth tones—beige, rust, milky white, pistachio, grey. Accessories: African-inspired jewelry, amulets, wide belts, sun hats, silk scarves, jute bags. Frequent animal prints—leopard, snake, zebra—and exotic leathers. Seasonal: warm climate, avoid with heavy winter outerwear. Dual use: vacation resort and city office depending on accessories. Quick ID: utility dress in beige with patch pockets + African jewelry or animal print.
**marine**  
Nautical theme: Romance of travel, yacht-club elegance, 1950s cruise mood. Straight or semi-fitted silhouettes, short shorts, cropped banana pants, relaxed V-neck sweaters. Key signs: horizontal stripe 2–3 cm, ropes, anchors, lifebuoys, yacht crests, gold “ship-wheel” buttons, metal eyelets. Core items: sailor stripe top, pea coat, white culottes, 7/8 chinos, silk scarf, topsiders. Accessories: braided rope belts, head scarves, mini drawstring backpack, aviator sunglasses, hoop earrings, simple bracelets. Footwear: topsiders, espadrilles, white sneakers, moccasins. Quick ID: stripe tee + gold buttons.
**jockey**  
Equestrian-inspired: Origin — British aristocracy, rider’s and hunting uniform of the 18th century. Modern wardrobe reference to equestrian attire: aristocratic, “expensive,” noble.
Palette: natural muted for everyday, light and dressy for competitions — e.g. brown, black, gray, khaki, blue, beige, white, burgundy. Fabrics: natural luxury — cashmere, silk, fine cotton, genuine leather. Allowed straight simple dresses (knit, leather, suede), straight skirts, shorts. Accessories reinforce the DNA: horse, horseshoe, bridle motifs; leather bracelets and belts; cap, fedora, baseball cap; gloves; classic bags; jokey boots! Brand reference — Hermès with equestrian aesthetic.
**dandy**  
Dandy style is built on basic pieces styled after menswear. It is not androgyny but emphasized femininity through accentuating and exaggerating masculine elements — sharp shoulders, straight cuts, ties, vests, masculine shoes — to create a bold yet feminine look. Not unisex: the image reads femininity. Not just base: dandy always "exaggerates" masculine elements. Dandies were marked by idle elegance, refined aesthetics, meticulous grooming, and a strict wardrobe. The total look should be laconic but expressive. The mood is sharp, confident, a little ironic, with an undertone of freedom and defiance.
**retro**  
Modern clothing styled after past decades (mainly 1920s, 1930s, 1940s, 1950s). Not originality, but recognizable era styling by silhouettes, prints, fabrics. 
Here you MUST specify the decade!
**drama**  
Theatrical effect, wow-factor, sexuality with provocation, luxury with aggression. Focus on power and attention, not comfort. Shape: armor-like silhouettes, sharp shoulders, tight waist, graphic hips, dramatic flare. Materials: leather, eco-leather, vinyl, wool suiting, mesh, lace, sequins, lycra. Decor: spikes, studs, big eyelets, straps, bold zippers, heavy buckles.Accessories: corset belts, chokers, ear cuffs, over-the-knee boots, pointed shoes, second-skin gloves. Colors: black dominates; accents in white, red, emerald, sapphire, gold, silver
Distinctions from military/safari: drama = glamour and nightlife; military/safari = function and nature. From avant-garde: drama = loud, sensual, rich; avant-garde = quiet, intellectual
**ethnic(**name of ethnicity**)**  
National motifs: african, indian, spanish (flamenco), indigenous, cowboy, japanese, chinese, ancient, egyptian, arab, slavic, mexican, etc.
**boho**  
Stems from the bohemian milieu (Romani → French creatives), then ’60s hippie; later, “boho-chic.” Essence: free self-expression, a multicultural mix, an inclusive and diverse vibe. Four pillars: loose silhouettes • layering • mixed prints/textures • ethnic motifs. Silhouette often read oversized and airy-flowing sleeves, ruffled/flounced skirts, tunics, off-shoulder peasant blouses, patchwork-like trousers, kimono, cowboy boots, felt boots, woven sole, gladiators, soft babouches, etc.; natural fibers (cotton, silk, linen, wool) plus corduroy, lace, macramé; suede/leather/fur (often faux) and denim, velvet flared. Texture play-rough with soft, smooth with nubby; embroidery and beads/glass beads; fringe as the clearest tell. Palette: neutrals to brights; prints: florals, paisley, tie-dye, ethnic, patchwork. Jewelry: noticeable, multi-layered, tassels, stones, wood, vintage gold/silver, handmade look.
**romantic**  
Naive, girlish, dreamy mood. Colors are pastel, soft, lightened (never black or bright red). Always ruffles, bows, drapery, pleats, layered skirts, bell or baby-doll shapes, tiering, puff/“ram’s horn” sleeves, narrow waist with voluminous skirt, etc. Fabrics are airy and flowing — silk, organza, lace, sometimes feathers or fur. Prints are childlike: flowers, hearts, butterflies, polka dots, cute or cartoon motifs. Creates an image that is tender and naive, not overtly sexual.
**feminine**  
Womanly lady-like style, elegance and adult confidence, figure-flattering. Complex cut with folds, drapery, waist and hip accents, moderate slits or necklines. Preferred lengths: knee, midi, maxi. Silhouettes fitted or semi-fitted. Prints include floral, watercolor, polka dots, geometry, check. Accessories are lady-like: pumps, sandals, fitted boots, small structured handbags. Jewelry is modern and refined, not retro. Overall image is noble, graceful, attractive without vulgarity, less “fairy-like” than Romantic and in contrast looks serious and focused on sensuality. Bags/Jewelry: modern refined (avoid retro “granny” vibe).
**conceptualism**  
Avant-garde style. Clothing as idea or manifesto. Unusual materials and forms; concept more important than practicality. May be sculptural or theatrical, not necessarily wearable, designed to provoke emotion: e.g Lady Gaga’s meat dress; Met Gala themes, Jeremy Scott’s fast-food looks.
**de-constructivism**  
Avant-garde style. About breaking construction: garments appear deliberately “wrong” or unfinished; rules of cut and proportion are disrupted. Key features: asymmetry, distorted proportions, oversized volumes; exposed darts and seams; intentional knee “bubbles”; inside-out finishes; hybrid “Frankenstein” pieces stitched from two garments. Traits extend to accessories. Low focus on sexuality; emphasizes comfort, versatility, creativity. 
**minimalism**  
“Simple but not simplistic” — a modern base with a touch of avant-garde. Maximum simplicity and “silence”: pure forms and reduced decoration. А clean cut and at least one avant-garde accent: asymmetry, unusual cutouts/slits, subtle proportion shifts, raw edges, exposed seams, streamlined silhouettes, structured forms, maxi length, architectural lines, color-blocking. Fabrics: wool, knits, suiting, denim, leather; matte or semi-matte, smooth textures, ribbed knits allowed. Palette: neutrals, pastels, brights/neons on simple shapes; monochrome, restrained checks/stripes, geometric motifs; accessories pared-down with twist (e.g.square-toe footwear). Focus on form itself. Cold intellectual minimalism.
**classic**  
Base. Casual style, but strict protocol clothing for officials: suit sets, closed dresses, calm palette (navy, grey, black, beige), minimal jewelry. Purpose — reliability and restraint, not fashion. timeless “reliable uniform”. Very rare. Qween family, etc.
**business-best**  
Base. Casual style, but Modern version of classic for conservative professions (diplomats, bankers, lawyers). Strict suits, restrained colors (navy, grey, burgundy, emerald), small checks or stripes allowed. Contemporary cuts without extremes. Goal — status and trust.
**business-casual**  
Base. Casual style, but Less strict business style: blazers, trousers, pencil skirts, blouses, soft suit fabrics. Neutral with muted accents. Allows modern silhouettes, textures and accessories. Professional but not rigid.
Common in conservative industries focused on money and reputation: finance, law, pharmaceuticals. Usually explicitly prescribed and followed at all levels, especially by employees with representative functions — the “face of the company.” Modern cuts combined with a system of restrictions: clothing should minimize fuss, inspire trust, and convey stability and reliability — not flashy success, but steady professionalism.
**smart-casual**  
Base. Casual style, but more appropriate for creative professions or without a strict dress code. Balances fashion, chic, and individuality, allowing more freedom of self-expression while keeping a polished look.
Relaxed business style with casual elements, e.g.: blazer+jeans, shirt+chinos, t-shirts+suit, suit+sneakers, simple dresses, slip dress + sweater, pleated midi skirt + tank top + linen shirt. Items combine comfort and respectability.
**city-casual** 
Everyday basic casual clothing — less formal than the above. Includes most categories except very formal items (e.g., blazers) e.g.: jeans, T-shirts, sweatshirts, sneakers and other garments once called "sportswear" but now worn casually. Focus on comfort, practicality and simplicity.
• `confidence` is a float **0–1** (0.75 = medium-sure).
• `category` → top-level product type label used for routing taxonomy and vocabularies of **META_CATEGORY_NAME**: **CATEGORY_EXAMPLES**.
• `model_construction` → category-specific canonical cut/shape/silhouette label if exist **MODEL_EXAMPLES** or сonstruction features not mentioned above.
• `cut_features`→ multi-tag field for **only** not mentioned above intentional **patternmaking** and **construction** techniques. Records stable design choices—**shaping**, **openings**, **paneling & seam placement**, **line direction/symmetry**, **edge finishes**, **fastening setup**, **internal supports**—сonstructions features that remain legible in wear and motion.

### Instructions
1.Input: Russian or English. Output enums: English.
2.If a required field is neither visible in the image nor mentioned in the description, leave it empty, don't make it up. Otherwise select the closest option.
3. Analyze the item's description and image. 
• `consistency_check` → `match` image shows the item named in the description; `mismatch` - attributes differ; `missing` - item absent; `cropped` - item is cropped in the image.
4. If match - take data from description, add attributes from image.
5. If missing or mismatch - take category from description, other attributes from image, but only for the described item.
'''

'''
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