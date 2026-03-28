"""Vector screen display — render emoji, text, and icons on the 184x96 face.

The screen module handles all rendering for Vector's LCD face. Content is
rendered via PIL and sent to the SDK's screen API as RGB565 image data.

Display types:
  - emoji: Full-colour Apple Color Emoji
  - text: Short text rendered in white on black
  - icon: Pre-defined simple icons drawn programmatically
  - color: Solid colour fill for mood indication
"""

from __future__ import annotations

import re
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Vector's screen dimensions.
SCREEN_WIDTH = 184
SCREEN_HEIGHT = 96

# Emoji regex for extraction from text.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

# Colour names to RGB.
COLOR_MAP = {
    "red": (255, 50, 50),
    "orange": (255, 150, 30),
    "yellow": (255, 230, 50),
    "green": (50, 220, 80),
    "cyan": (50, 220, 220),
    "blue": (50, 100, 255),
    "purple": (180, 80, 255),
    "pink": (255, 120, 180),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def extract_emoji(text: str) -> tuple[str, str]:
    """Extract emoji from text. Returns (clean_text, emoji_string)."""
    emojis = "".join(_EMOJI_PATTERN.findall(text))
    clean = _EMOJI_PATTERN.sub("", text).strip()
    return clean, emojis


def _load_emoji_font(size: int = 64) -> ImageFont.FreeTypeFont | None:
    """Try to load a font that supports colour emoji."""
    for path in [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return None


def _load_text_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Load a clean text font."""
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFCompact.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_emoji(emoji: str) -> Image.Image:
    """Render emoji as a full-colour image for Vector's screen."""
    img = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_emoji_font(64)
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), emoji, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (SCREEN_WIDTH - tw) // 2
    y = (SCREEN_HEIGHT - th) // 2
    draw.text((x, y), emoji, font=font, embedded_color=True)
    return img


def render_text(text: str, color: str = "white") -> Image.Image:
    """Render short text on Vector's screen."""
    img = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    rgb = COLOR_MAP.get(color, (255, 255, 255))

    # Auto-size: try large first, shrink if needed.
    for size in [28, 22, 16, 12]:
        font = _load_text_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        if tw <= SCREEN_WIDTH - 10:
            break

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (SCREEN_WIDTH - tw) // 2
    y = (SCREEN_HEIGHT - th) // 2
    draw.text((x, y), text, fill=rgb, font=font)
    return img


def render_color(color: str) -> Image.Image:
    """Render a solid colour fill."""
    rgb = COLOR_MAP.get(color, (0, 0, 0))
    return Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), rgb)


def render_icon(name: str) -> Image.Image:
    """Render a simple programmatic icon."""
    img = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    if name == "heart":
        # Simple heart shape (flip y so point is at bottom).
        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                nx = (x - cx) / 35
                ny = -(y - cy - 5) / 35
                if (nx**2 + ny**2 - 1)**3 - nx**2 * ny**3 <= 0:
                    img.putpixel((x, y), (255, 50, 80))

    elif name == "star":
        import math
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = 38 if i % 2 == 0 else 18
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=(255, 220, 50))

    elif name == "sun":
        draw.ellipse([cx-25, cy-25, cx+25, cy+25], fill=(255, 220, 50))
        import math
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = cx + 30 * math.cos(angle)
            y1 = cy + 30 * math.sin(angle)
            x2 = cx + 42 * math.cos(angle)
            y2 = cy + 42 * math.sin(angle)
            draw.line([(x1, y1), (x2, y2)], fill=(255, 220, 50), width=3)

    elif name == "moon":
        draw.ellipse([cx-30, cy-30, cx+30, cy+30], fill=(240, 230, 180))
        draw.ellipse([cx-15, cy-35, cx+25, cy+25], fill=(0, 0, 0))

    elif name == "question":
        font = _load_text_font(60)
        bbox = draw.textbbox((0, 0), "?", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((SCREEN_WIDTH-tw)//2, (SCREEN_HEIGHT-th)//2), "?", fill=(100, 200, 255), font=font)

    elif name == "exclamation":
        font = _load_text_font(60)
        bbox = draw.textbbox((0, 0), "!", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((SCREEN_WIDTH-tw)//2, (SCREEN_HEIGHT-th)//2), "!", fill=(255, 80, 80), font=font)

    elif name == "checkmark":
        draw.line([(cx-25, cy), (cx-8, cy+20), (cx+30, cy-25)], fill=(80, 255, 80), width=6)

    elif name == "x_mark":
        draw.line([(cx-25, cy-25), (cx+25, cy+25)], fill=(255, 80, 80), width=6)
        draw.line([(cx+25, cy-25), (cx-25, cy+25)], fill=(255, 80, 80), width=6)

    elif name == "zzz":
        font = _load_text_font(36)
        draw.text((cx-30, cy-30), "Z", fill=(150, 150, 255), font=font)
        font2 = _load_text_font(24)
        draw.text((cx, cy-5), "z", fill=(120, 120, 220), font=font2)
        font3 = _load_text_font(16)
        draw.text((cx+20, cy+15), "z", fill=(90, 90, 180), font=font3)

    else:
        # Unknown icon — render as text.
        return render_text(name)

    return img


# All available icon names for the tool enum.
ICON_NAMES = [
    "heart", "star", "sun", "moon", "question",
    "exclamation", "checkmark", "x_mark", "zzz",
]
