@@ -0,0 +1,52 @@
import streamlit as st
from PIL import Image
import io
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="faster but less creative", layout="wide")
st.title("👕 Kevin 2 版本  (faster but less creative)")

@st.cache_resource(show_spinner=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# 👕 New options
shirt_color = st.radio("Select shirt color:", ["Black", "White"])
shirt_gender = st.radio("Select shirt gender:", ["Men", "Women"])

uploaded_files = st.file_uploader("Upload T-shirt designs (images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def generate_title(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return f"{shirt_gender}'s Pure - Cotton {shirt_color} T - shirt: \"{caption}\""

if uploaded_files:
    results = []
    with st.spinner("Generating titles locally. Please wait..."):
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            try:
                title = generate_title(img)
                results.append(title)
            except Exception as e:
                results.append(f"ERROR: {e}")

    st.success("Done! Copy or export your results below.")
    full_text = "\n".join(results)
    st.text_area("Generated Titles (Excel-ready)", full_text, height=300)

    df = pd.DataFrame({"Title": results})
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    st.download_button("Download as Excel", data=excel_buffer, file_name="tshirt_titles_offline.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
