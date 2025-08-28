import torch
import sys
import os

def export_onnx_from_torchscript():
    if len(sys.argv) != 6:
        print("参数错误！正确格式：")
        print("python export_onnx_torchscript.py <TorchScript模型路径> <ONNX导出路径> <输入名称> <输出名称> <输入维度>")
        sys.exit(1)

    # 解析参数
    ts_model_path = sys.argv[1]
    onnx_path = sys.argv[2]
    input_name = sys.argv[3]
    output_name = sys.argv[4]
    input_dim = int(sys.argv[5])

    try:
        # 直接加载C++导出的TorchScript模型（含结构+权重）
        model = torch.jit.load(ts_model_path, map_location="cpu")
        model.eval()  # 切换到推理模式

        # 创建示例输入
        dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

        # 导出ONNX（直接使用加载的模型，无需手动定义结构）
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=onnx_path,
            opset_version=12,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"}
            }
        )

        if os.path.exists(onnx_path):
            print(f"成功：ONNX已导出至 {onnx_path}")
            sys.exit(0)
        else:
            raise FileNotFoundError("ONNX文件未生成")

    except Exception as e:
        print(f"失败：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    export_onnx_from_torchscript()
