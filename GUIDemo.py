import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from Task2 import Vgg16_net
from torchvision import transforms

# 定义转换：调整大小和转换为张量
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 加载模型
model = Vgg16_net()
model.load_state_dict(torch.load('save/Task2/model_epoch_29.pth'))
model.eval()

# 启动摄像头并进行预测
def start_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 图像转换为 PIL 图像
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 应用转换
        input_tensor = transform(pil_image)
        
        # 添加批处理维度
        input_tensor = input_tensor.unsqueeze(0)

        # 进行预测
        with torch.no_grad():
            model.eval()
            prediction = model(input_tensor)

        print(prediction)


        # 将捕获的帧转换为模型可以处理的格式，并进行预测
        # 注意：需要根据模型的输入要求来调整预处理步骤
        # ...

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 创建主窗口
window = tk.Tk()
window.title("Image Classifier")

# 创建按钮
btn = tk.Button(window, text="Start Camera", command=start_camera)
btn.pack()

# 运行事件循环
window.mainloop()