import onnxruntime as ort
import numpy as np

MODEL_PATH = r"C:\Users\jsbow\src\wallballdoll\assets\model\pose_model.onnx"

def test_model():
    print(f"Loading model: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    
    print("\n=== Model Info ===")
    meta = session.get_modelmeta()
    print(f"IR version: {getattr(meta, 'ir_version', 'N/A')}")
    print(f"Producer: {getattr(meta, 'producer_name', 'N/A')}")
    print(f"Version: {getattr(meta, 'version', 'N/A')}")
    print(f"Description: {getattr(meta, 'description', 'N/A')}")
    
    print("\n=== Inputs ===")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
    
    print("\n=== Outputs ===")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")
    
    # Create dummy input based on first input's shape
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    
    # Handle dynamic dimensions (replace -1 or None with 1)
    concrete_shape = []
    for dim in input_shape:
        if isinstance(dim, str) or dim is None or dim <= 0:
            concrete_shape.append(1)
        else:
            concrete_shape.append(dim)
    
    print(f"\n=== Running inference with dummy input ===")
    print(f"Input shape: {concrete_shape}")
    
    dummy_input = np.random.randn(*concrete_shape).astype(np.float32)
    input_name = input_info.name
    
    outputs = session.run(None, {input_name: dummy_input})
    
    print(f"\n=== Outputs ===")
    for i, out in enumerate(outputs):
        out_info = session.get_outputs()[i]
        print(f"  {out_info.name}: shape={out.shape}, dtype={out.dtype}")
        print(f"    min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
    
    print("\n[OK] Model loaded and inference successful!")

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()