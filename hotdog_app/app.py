from fastai.vision.all import *
import gradio as gr

def is_hotdog(x): return x[0].isupper()

learn = load_learner('hotdog.pkl')

categories = ('hotdog', 'pasta', 'pizza', 'salad', 'sandwich')

def classify_images(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs))) #gradio interface expects a dictionary

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = [f'{c}.jpg' for c in categories]


intf = gr.Interface(fn=classify_images, inputs=image, outputs=label, examples=examples)
intf.launch()