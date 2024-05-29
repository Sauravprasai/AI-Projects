import streamlit as st
import ctransformers


#Loading the model from local device
@st.cache_resource()
def load_model_text_llama2_gguf(model_path, model_file, model_type="llama", gpu_layers=50):
    model = ctransformers.AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=model_path,
                                                               model_file=model_file,
                                                               model_type=model_type, gpu_layers=gpu_layers)
    return model


#Downloading and Adding Model to the local device
def generate_text_llama2_gguf(model, prompt, text_area_placeholder, stop=["\n", "Question:", "Q:"]):
    generated_text = ""
    for text in model(f"Question: {prompt} Answer:", stream=True, stop=stop):
        generated_text += text
        text_area_placeholder.markdown(generated_text, unsafe_allow_html=True)
    return generated_text


def main():
    st.title("Text Generation using Lama2 GGUF Model")

    model_path = "../Models/models--TheBloke--Llama-2-7B-GGUF/snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80"
    model_file = "llama-2-7b.Q4_K_M.gguf"
    model = load_model_text_llama2_gguf(model_path, model_file)
    prompt = st.text_input("Enter your prompt", value="Write an essay on quantum computing")
    text_area_placeholder = st.empty()
    if st.button("Generate Text"):
        result = generate_text_llama2_gguf(model, prompt, text_area_placeholder)


if __name__ == '__main__':
    main()
