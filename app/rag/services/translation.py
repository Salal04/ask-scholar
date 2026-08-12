from ..gemini_manager import key_manager

def translate(text, target_lang):
    TRANSLATION_PROMPT = f"""
Text:
{text}

Translate the above text into {target_lang}.

Rules:
- Output only the translation.
- Do not explain or summarize.
- Preserve the original meaning.
- Preserve names, numbers, and important terminology.
- Do not add any information.
"""

    def fn(client, model):
        response = client.models.generate_content(
            model=model,
            contents=TRANSLATION_PROMPT,
        )
        return response.text.strip()

    return key_manager.call("translation", fn)