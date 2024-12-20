import streamlit as st
from PIL import Image
import io
import base64

#SUPPRESS TEXT OUTPUT
import sys
sys.stdout=io.StringIO()
import numpy as np
#SUPPRESS WARNINGS
import warnings
warnings.filterwarnings("ignore")

#IMPORT PACKAGES
from functools import lru_cache
import tensorflow as tf
import tensorflow_hub as hub

#FUNCTION - CENTRED SQUARE CROP
def crop(image):
  _=image.shape
  height,width=_[1],_[2]
  _=min(height,width)
  target_height,target_width=_,_
  offset_height=max(height-width,0)//2
  offset_width=max(width-height,0)//2
  image_crop=tf.image.crop_to_bounding_box(
      image,offset_height,offset_width,target_height,target_width)
  return image_crop

#FUNCTION - LOAD IMAGE
@lru_cache(maxsize=None)
def load_image(source):
    #DOWNLOAD (IF URL)
    _=source
   
    #DECODE
    image=tf.io.decode_image(tf.io.read_file(_),channels=3,dtype=tf.float32)
   
        
    #BATCH DIMENSION
    image=image[tf.newaxis,...]
    #CENTRED SQUARE CROP
    image=crop(image)
    #RESIZE
    image=tf.image.resize(image,(256,256))
    return image

model = 5


def preprocess(content, style_num = 1):
  print('ouptut generated')
  global model
  print('ouptut generated')
  if model == 5:
    _="https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    model=hub.load(_)
    print('ouptut generated2')

  style=load_image(f"{style_num}.jpg")
  print('ouptut generated3')
    
  generated = model(tf.constant(content),tf.constant(style))[0]
  print('ouptut generated4')
  return style, generated
  




# Streamlit app
st.set_page_config(
    page_title="Neural Style Transfer App",
    layout="wide",
    page_icon="🖼",
    initial_sidebar_state="expanded"
)

st.title("🎨 Neural Style Transfer App")

st.markdown(
    """
    This app applies **Neural Style Transfer** to your uploaded image. 
    Upload an image to see the magic of AI-powered artistic transformations!
    """
)

# Sidebar for instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown(
        """
        1. Upload an image in PNG, JPG, or JPEG format.
        2. The app will process the image using Neural Style Transfer.
        3. View the original and stylized images side by side.
        4. Download one of the stylized images if you like it!
        """
    )

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    style_num = st.selectbox("Select Style", ["Style 1", "Style 2", "Style 3", "Style 4", "Style 5", "Style 6", "Style 7", "Style 8", "Style 9"])
    # Load the image
    input_image = Image.open(uploaded_file)
    input_image.save('content.jpg')

    content_image = load_image('content.jpg')
    

    # Process the image
    print('ouptut generated')
    with st.spinner("Applying Neural Style Transfer... Please wait!"):
        style, generated = preprocess(content_image, style_num)

    # Display images side by side
    st.subheader("Results")
    st.markdown("### Original and Stylized Images")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(content_image.numpy(), caption="Original Image", use_column_width=True)

    with col2:
        st.image(style.numpy(), caption="Stylized Image 1", use_column_width=True)

    with col3:
        st.image(generated.numpy(), caption="Generated Image", use_column_width=True)

    # Make one of the processed images available for download
    st.markdown("### Download Your Image")
    buffered = io.BytesIO()
    Image.fromarray(np.uint8(generated.numpy()[0]*255)).resize((1024, 1024)).save(buffered, format="PNG")
    buffered.seek(0)
    b64 = base64.b64encode(buffered.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="stylized_image2.png" style="color: white; text-decoration: none; background-color: #4CAF50; padding: 10px 20px; border-radius: 5px;">Download Stylized Image 2</a>'

    st.markdown(href, unsafe_allow_html=True)
else:
    st.info("Please upload an image to get started!")

# Footer
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .css-1q1n0ol {
        margin-top: 50px;
        text-align: center;
    }
    </style>
    <div class="css-1q1n0ol">Agaba_Embedded</div>
    """,
    unsafe_allow_html=True
)
