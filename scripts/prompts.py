"""Headshot prompt library. Each entry is a format string with a {trigger}
placeholder. generate_headshots.py substitutes the trigger word at runtime."""

from __future__ import annotations

HEADSHOT_PROMPTS: list[str] = [
    # Classic corporate / executive
    "professional corporate headshot of {trigger}, navy business suit, crisp "
    "white shirt, soft studio lighting, neutral gray seamless background, "
    "sharp focus on eyes, shallow depth of field, shot on 85mm f/1.4, "
    "confident expression, color grade, photorealistic",

    # Editorial magazine cover
    "editorial magazine portrait of {trigger}, black turtleneck, minimalist "
    "white background, dramatic rim lighting from the left, high contrast, "
    "sharp detail, shot on medium format, hasselblad aesthetic, photorealistic",

    # Environmental office
    "environmental executive headshot of {trigger}, charcoal blazer over "
    "white shirt, modern office background with soft bokeh, warm natural "
    "window light from the side, 50mm lens, looking directly at camera, "
    "photorealistic",

    # Approachable LinkedIn
    "friendly linkedin headshot of {trigger}, light blue oxford shirt, "
    "outdoor blurred greenery background, golden hour sunlight, subtle "
    "genuine smile, shot on 85mm, warm color grade, photorealistic",

    # Tech founder vibe
    "tech founder portrait of {trigger}, dark gray hoodie, clean white "
    "background, bright even lighting, centered composition, approachable "
    "expression, shot on 50mm f/1.8, photorealistic",

    # Moody creative director
    "creative director portrait of {trigger}, black button up shirt, moody "
    "dark gray background, single key light from the right, cinematic color "
    "grade, intense confident expression, shot on 85mm, photorealistic",

    # Speaker headshot
    "conference speaker headshot of {trigger}, burgundy blazer, slightly "
    "blurred stage background, warm spotlight from above, confident smile, "
    "shot on 85mm f/1.8, photorealistic",

    # Outdoor natural
    "outdoor natural portrait of {trigger}, olive field jacket, blurred "
    "mountain background, soft overcast light, relaxed neutral expression, "
    "shot on 85mm, natural color grade, photorealistic",

    # Black and white fine art
    "black and white fine art portrait of {trigger}, dark shirt, studio "
    "black background, rembrandt lighting, film grain, shot on medium "
    "format, classic headshot composition",

    # Coworking casual
    "relaxed coworking portrait of {trigger}, heather gray t-shirt, out of "
    "focus brick wall background, soft window light, genuine laugh, shot on "
    "50mm f/1.4, photorealistic",

    # Boardroom formal
    "formal boardroom portrait of {trigger}, dark navy three piece suit, "
    "dark wood paneled background, warm tungsten key light, serious "
    "confident expression, shot on 85mm f/2, photorealistic",

    # Clean white headshot
    "clean studio headshot of {trigger}, white dress shirt, pure white "
    "seamless background, soft diffused lighting from both sides, neutral "
    "expression, sharp focus, shot on 85mm f/2.8, photorealistic",
]
