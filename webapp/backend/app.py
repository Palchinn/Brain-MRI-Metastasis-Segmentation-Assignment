import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from models.nested_unet import NestedUNet  # or AttentionUNet

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NestedUNet(num_classes=1, input_channels=1).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Convert the prediction to a binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Convert the binary mask to a base64 encoded string
    buffered = io.BytesIO()
    Image.fromarray(binary_mask).save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse(content={"mask": mask_base64})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)