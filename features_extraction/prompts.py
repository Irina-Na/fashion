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
CLASSIC
• Idea & Mood: Formal neutrality; timeless “reliable uniform”.
• Silhouette/Cut: Tailored, straight lines, jacket or knee‑length skirt.
• Materials/Texture: Worsted wool, twill, crisp shirting.
• Colours/Prints: Dark navy, charcoal, black, touches of white/beige; solid.
• Unique Markers: Notch lapel, short button stance, pearl studs, sleek grooming.

BUSINESS‑BEST
• Idea & Mood: Strict, conservative corporate dress code.
• Silhouette/Cut: Fitted trouser or pencil‑skirt suit, defined waist.
• Materials: Super 120s wool, smooth cotton shirting.
• Colours: Dark neutrals plus deep jewel accents (burgundy, emerald).
• Markers: Closed‑toe pumps ≤5 cm, silk pocket square, minimal adornment.

BUSINESS‑CASUAL
• Idea & Mood: “Office without a tie” – status + comfort.
• Silhouette: Semi‑fitted blazer with chinos or midi skirt.
• Materials: Blended suiting, jersey knits.
• Colours: Muted basics with a soft accent (sky blue, muted red).
• Markers: Clean loafers, subtle check/stripe, slim belt.

SMART‑CASUAL
• Idea & Mood: Polished yet relaxed city look.
• Silhouette: Structured blazer over dark‑wash denim or culottes.
• Materials: Fine cotton suiting, premium denim.
• Colours: Neutral palette + single subdued accent.
• Markers: Quality sneakers or Chelsea boots, leather backpack‑tote.

CASUAL (BASE)
• Idea & Mood: Functional modern basics.
• Silhouette: Simple straight forms.
• Materials: Cotton jersey, fleece, denim.
• Colours: Core solids.
• Markers: Minimal detail; designed to combine with anything.

SAFARI
• Idea & Mood: Adventure, hot‑climate colonial chic.
• Silhouette: Belted 4‑pocket jacket, shorts or banana trousers.
• Materials: Cotton twill, linen.
• Colours: Sand, khaki, olive.
• Markers: Epaulettes, wide belt, straw hat, wooden ethnic jewellery.

MILITARY
• Idea & Mood: Discipline, utilitarianism.
• Silhouette: Straight or oversized with emphasised shoulders.
• Materials: Rugged cotton, drill, serge.
• Colours: Khaki, olive, camouflage.
• Markers: Patch pockets, epaulettes, D‑rings, combat boots.

MARINE
• Idea & Mood: Nautical freshness, retro cruise.
• Silhouette: Semi‑fitted, pea coat, Breton top.
• Materials: Cotton, wool.
• Colours: Navy‑white‑red with gold buttons.
• Markers: Horizontal stripe, anchors, deck shoes.

DRAMA
• Idea & Mood: Wow‑effect, provocative sexuality.
• Silhouette: Corset tops, exaggerated shoulders or flared shapes.
• Materials: Leather, vinyl, sequins, latex.
• Colours: Black plus jewel tones.
• Markers: Spikes, chains, eyelets, high slits, thigh‑high “second‑skin” boots.

ROMANTIC
• Idea & Mood: Softness, naivety, airiness.
• Silhouette: Loose A‑lines, puff sleeves.
• Materials: Silk, organza, lace.
• Colours: Pastels, small florals, polka dots.
• Markers: Ruffles, bows, pleats, thin belt.

FEMININE
• Idea & Mood: Elegant, grown‑up femininity.
• Silhouette: Fitted blazer or sheath dress, midi length.
• Materials: Wool, silk, fine‑gauge knit.
• Colours: Any sophisticated hue, floral prints.
• Markers: Delicate heels, waist emphasis, gloves, refined jewellery.

JOCKEY
• Idea & Mood: English aristocratic equestrian sport.
• Silhouette: Slim breeches + cropped riding jacket.
• Materials: Wool, leather.
• Colours: Brown, black, burgundy.
• Markers: High riding boots with straps, kepi helmet, leather gloves.

DANDY
• Idea & Mood: Masculine tailoring with refinement.
• Silhouette: Menswear suit cut adapted to female figure.
• Materials: Wool, tweed.
• Colours: Grey, black, navy.
• Markers: Waistcoat, tie, loafer/oxford shoes, pocket‑watch chain.

RETRO
• Idea & Mood: Explicit quotation of the 1920s‑1950s.
• Silhouette: Era signatures (drop waist, New Look).
• Materials: Silk, brocade, nylon, corsetry fabrics.
• Colours/Prints: Period prints (1950s polka dot, Art Deco motifs).
• Markers: Midi length, corset + full skirt, pillbox hat, gloves.

ETHNIC
• Idea & Mood: Direct citation of national costume.
• Silhouette: A‑lines, ponchos, tunics.
• Materials: Cotton, wool, heavy embroidery or beadwork.
• Colours/Prints: Rich palette, traditional ornament.
• Markers: Fringe, woven belts, chunky wooden/beaded jewellery.

BOHO
• Idea & Mood: Free, bohemian ethnic mix.
• Silhouette: Oversize, layered.
• Materials: Linen, denim, macramé, suede.
• Colours: Earthy plus bright accents, tie‑dye.
• Markers: Fringe, embroidery, straw hats, cowboy ankle boots.

AVANT‑GARDE — MINIMALISM
• Idea & Mood: “Simple but not simple”: laconic + asymmetry.
• Silhouette: Clean volumes, architectural lines.
• Materials: Smooth monochrome fabrics.
• Colours: Neutrals (black/white, beige, graphite).
• Markers: Single proportion‑breaking accent, zero decoration.

AVANT‑GARDE — DE‑CONSTRUCTIVISM
• Idea & Mood: Broken construction, “Frankenstein garment”.
• Silhouette: Asymmetry, exposed seams, hyper volume.
• Materials: Wool, denim, mixed textiles.
• Colours: Dark neutrals.
• Markers: Lining on the outside, darts visible, spliced of two garments.

AVANT‑GARDE — CONCEPTUALISM
• Idea & Mood: “Speaking clothes”, shock art fashion.
• Silhouette: Any form dictated by concept.
• Materials: Any, including non‑traditional (meat, plastic).
• Colours: Chosen individually for the idea.
• Markers: Runway or performance piece; meaning outweighs wearability.

### Instructions
1. Use **only** the exact enum values listed above, exclude fabric.
2. Never invent new keys; every key must exist in the provided schema.  
3. Language of input may be Russian or English; output enums are **always English**. 
4. If a value cannot be inferred with reasonable certainty, output:
  – `null`   for scalars,  
  – `[]`     for lists. 
'''