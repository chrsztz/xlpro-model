# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import shutil
import os
from predict_fingering import predict_fingering
from io import BytesIO

app = FastAPI()

@app.post("/predict")
async def predict_fingering_endpoint(file: UploadFile = File(...)):
    # 验证文件类型
    if file.content_type != 'application/vnd.recordare.musicxml':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an MXL file.")

    # 保存上传的文件到临时目录
    temp_input_path = f"/tmp/{file.filename}"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 预测指法
        output_score = predict_fingering(temp_input_path)

        # 保存修改后的乐谱到临时输出路径
        temp_output_path = f"/tmp/modified_{file.filename}"
        output_score.write('mxl', fp=temp_output_path)

        # 读取修改后的文件并准备返回
        with open(temp_output_path, "rb") as f:
            modified_file = BytesIO(f.read())

        # 删除临时文件
        os.remove(temp_input_path)
        os.remove(temp_output_path)

        # 返回修改后的MXL文件
        return StreamingResponse(modified_file, media_type="application/vnd.recordare.musicxml", headers={"Content-Disposition": f"attachment; filename=modified_{file.filename}"})

    except Exception as e:
        # 删除临时文件
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
