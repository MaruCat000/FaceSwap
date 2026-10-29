"""カメラ診断: どの番号・方式で映像が取れるかを調べる。  実行: python cam_test.py"""
import time

import cv2

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]

for idx in range(4):
    for name, be in BACKENDS:
        cap = cv2.VideoCapture(idx, be)
        if not cap.isOpened():
            print(f"番号{idx} {name}: 開けない")
            cap.release()
            continue
        frame, got = None, 0
        t0 = time.time()
        for _ in range(30):              # 最初の数枚は暗いことがあるので30枚読む
            ok, f = cap.read()
            if ok:
                frame, got = f, got + 1
        dt = max(time.time() - t0, 1e-6)
        if frame is None:
            print(f"番号{idx} {name}: 開けるが映像が取れない")
        else:
            m = frame.mean()
            verdict = "★映像あり" if m > 10 else "真っ黒(使えない)"
            print(f"番号{idx} {name}: {frame.shape[1]}x{frame.shape[0]} "
                  f"明るさ平均={m:.0f} {got}/30枚 {got / dt:.1f}fps  {verdict}")
        cap.release()
print("★が付いた 番号 と 方式 を、face_swap_mesh.py の CAM_INDEX / CAM_BACKEND に設定してください。")
