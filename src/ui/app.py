"""
Streamlit UI for the n-gram next-word predictor.
Runs alongside the CLI (main.py) — does not replace it.

Run with: streamlit run src/ui/app.py
"""

import os
import sys
from dotenv import load_dotenv
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


class PredictorUI:
    """
    Wraps a Predictor instance and renders a simple Streamlit web page:
    a text box for input, a slider for k, and a live display of the
    top-k predicted next words.
    """
    def __init__(self, predictor):
        """
        function: store the Predictor instance this UI will use
        parameters: predictor
        return: none
        """
        self.predictor = predictor
    def render(self):
        """
        function: draw the Streamlit page - title, text input, k slider,
                   and the resulting predictions
        parameters: none
        return: none
        """
        st.title("N-Gram Next-Word Predictor")
        st.write("Type a partial sentence and see the top-k predicted next words.")

        text = st.text_input("Your text:", "")
        k = st.slider("Number of predictions (k):", min_value=1, max_value=10, value=3)

        if text.strip() != "":
            predictions = self.predictor.predict_next(text, k)
            st.subheader("Predictions:")
            for word in predictions:
                st.write(f"- {word}")
        else:
            st.write("Start typing above to see predictions.")

def main():
    """
    function: load .env, build Normalizer + NGramModel + Predictor,
               then hand off to PredictorUI to render the page
    parameters: none
    return: none
    """
    load_dotenv("config/.env")

    model = NGramModel()
    model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)

    ui = PredictorUI(predictor)
    ui.render()

if __name__ == "__main__":
    main()

#streamlit run src/ui/app.py