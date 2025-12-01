import pytesseract
from PIL import Image
import google.generativeai as genai
import json
import matplotlib.pyplot as plt
import re
import cv2 as cv



def extract_text_from_image(img):
    text = pytesseract.image_to_string(img, lang='eng')
    return text

def structure_with_gemini(extracted_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are an expert invoice data extraction system. Analyze the following raw invoice text.
    Your task is to extract all line-item details, calculate the total item count, and
    determine the final reconciled amount (sum of all item_amount values).

    **INSTRUCTIONS:**
    1.  Assume the entire text belongs to 'page_no': '1'.
    2.  Extract the item name (Particulars), quantity (Qty.), rate (Rate), and amount (Amount)
        for every line item in the bill's table. If quantity or rate are not mentioned, use 0.0
    3.  Set 'total_item_count' to the count of unique line items found.
    4.  Set 'reconciled_amount' to the sum of all extracted 'item_amount' values.
    5.  The final output MUST strictly adhere to the provided JSON schema.
    6.  Include a 'token_usage' object with 'total_tokens', 'input_tokens', and 'output_tokens'.

    **REQUIRED JSON SCHEMA:**
    The final output must be a single JSON object (the response body for the API):
    {{
      "is_success": true,
      "data": {{
        "pagewise_line_items": [
          {{
            "page_no": "1",
            "bill_items": [
              {{
                "item_name": "...",
                "item_amount": 0.00,
                "item_rate": 0.00,
                "item_quantity": 0.00
              }}
              // ... continue for all line items
            ]
          }}
        ],
        "total_item_count": 0,
        "reconciled_amount": 0.00,
        "token_usage": {{
          "total_tokens": 0,
          "input_tokens": 0,
          "output_tokens": 0
        }}
      }}
    }}

    Invoice text:
    ---
    {extracted_text}
    ---

    Return ONLY the valid JSON object, without any surrounding markdown, backticks, or explanation. The output should strictly not contain any characters that are not part of the json syntax
    """



    response = model.generate_content(prompt)
    return response

def format_response(response):
    candidate = response.candidates[0]
    content = candidate.content
    raw_json_string = content.parts[0].text
    cleaned_json_string = re.sub(r'```json|```', '', raw_json_string, flags=re.IGNORECASE).strip()
    final_json_data = json.loads(cleaned_json_string)
    
    # Extract actual token usage from Gemini response
    if hasattr(response, 'usage_metadata'):
        token_usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count
        }
        # Update the token_usage in the response data
        if 'data' in final_json_data and 'token_usage' in final_json_data['data']:
            final_json_data['data']['token_usage'] = token_usage
    
    return final_json_data  # Return as formatted JSON string
