# image_summarizer.py
import base64
from openai import OpenAI

class ImageSummarizer:
    SYSTEM_PROMPT = """You are a financial document analyst. 
    Describe this image extracted from a banking PDF in precise detail.
    Focus on: charts, graphs, organizational structures, data visualizations.
    Output a rich text summary that can be used for semantic search retrieval."""

    def __init__(self):
        self.client = OpenAI()

    def summarize(self, image_path: str, context: str = "") -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text",
                     "text": f"Context from surrounding text: {context}\n\n{self.SYSTEM_PROMPT}"}
                ]
            }],
            max_tokens=500
        )
        return response.choices[0].message.content