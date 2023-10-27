from pathlib import Path
import streamlit as st
import rengine 
import base64
from PIL import Image, ImageDraw, ImageFont
import json


TINT_COLOR = (0, 0, 0) 
TRANSPARENCY = .1  
OPACITY = int(255 * TRANSPARENCY)

upload_path = 'uploads/test.png'
threshold = 0.53
font = ImageFont.truetype("font/font.ttf")

st.sidebar.image("static/img.jpg",use_column_width=True)
st.sidebar.title("CAR DAMAGE DETECTION")

st.header("Project UI for live demo")
st.info("please use a small image for faster execution")
st.write("please upload an image of a car with damage/no damage")

st.subheader("select file")
file = st.file_uploader("Jpeg or Png",type=['png','jpg','jpeg'])
if file is not None:
    st.image(file)
    img = Image.open(file)
    if img.mode in ("RGBA", "P"):
        im = img.convert('RGB')
        im.save(upload_path)
    img.save(upload_path)
    st.success("image uploaded")
    pb = st.progress(20)
    with st.spinner("running AI models, please wait"):
        data = rengine.predict(upload_path)
        pb.progress(50)
        if data is not None:
            r = json.loads(data)
            pb.progress(60)

            if r.get("message","Failed") == "Success":
                results = r.get("result",None)
                if results:
                    for item in results:
                        if item.get("message","Failed") == "Success":
                            preds = item.get("prediction")
                            if preds:
                                for i,row in enumerate(preds):
                                    if row.get("score") >= threshold:
                                        overlay = Image.new('RGBA', img.size, TINT_COLOR+(0,))
                                        draw = ImageDraw.Draw(overlay)
                                        xy = [row['xmin'],row['ymin'],row['xmax'],row['ymax'],]
                                        draw.rectangle(xy,fill=(255,255,255,4), outline=(0,0,0,), width=2 )
                                        pos = (row['xmin']+10,row['ymin']+10)
                                        draw.text(pos, row.get("label"), fill=255, font=font, )
                                        del draw
                                    if i == 3:
                                        pb.progress(100)
                                        break
                                else:
                                    pb.progress(100)
                                img = img.convert('RGBA')
                                img = Image.blend(img, overlay,.4)
                                st.header("RESULT")
                                st.write("damage found")
                                st.image(img.resize((int(img.width*1.5),int(img.height*1.5)), Image.LINEAR))
                            else:
                                pb.progress(100)
                                st.header("RESULT")
                                st.write("no damage in this vehicle")
                                st.image(img.resize((int(img.width*1.5),int(img.height*1.5)), Image.LINEAR))
                    else:
                        st.success("prediction done")
            else:
                pb.progress(100)
                st.error("error, could not predict in this image, Try another")
                st.write(r)
        else:
            st.error("could not load result")
        
            



