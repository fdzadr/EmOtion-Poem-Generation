import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer_emotion.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('model_emotion.h5')

def generate_poem(seed_text, next_words):
    poem = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=130-1)
        
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=-1).item()
        
        output_word = tokenizer.index_word[predicted]
        
        seed_text += " " + output_word
        poem += " " + output_word

    return poem

st.title("EmOtion")

st.subheader("EmOtion : Poem Generator yang dapat membuat 1 bait puisi (1 bait puisi = 4 baris puisi)")

st.write("*Note : Input is in English only!")

seed_text_input = st.text_input("Masukkan kata:")

next_words = st.number_input("Jumlah kata selanjutnya:", min_value=1, max_value=100, value=30)

if st.button("Generate Poem"):
    generated_poem = generate_poem(seed_text_input, next_words)
    st.write("Puisi yang dihasilkan:")
    st.write(generated_poem)

    last_word_generated_poem_1 = " ".join(generated_poem.split()[-2:])
    st.write("Puisi yang dihasilkan (Baris Kedua):")
    seed_text_input_2 = last_word_generated_poem_1
    generated_poem_2 = generate_poem(seed_text_input_2, next_words)
    st.write(generated_poem_2)

    last_word_generated_poem_2 = " ".join(generated_poem_2.split()[-2:])
    st.write("Puisi yang dihasilkan (Baris Ketiga):")
    seed_text_input_3 = last_word_generated_poem_2
    generated_poem_3 = generate_poem(seed_text_input_3, next_words)
    st.write(generated_poem_3)

    last_word_generated_poem_3 = " ".join(generated_poem_3.split()[-2:])
    st.write("Puisi yang dihasilkan (Baris Keempat):")
    seed_text_input_4 = last_word_generated_poem_3
    generated_poem_4 = generate_poem(seed_text_input_4, next_words)
    st.write(generated_poem_4)
