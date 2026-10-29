"""
MediaPipe Face Mesh + 三角形ワープによる簡易リアルタイム顔入れ替え(CPU向け)

準備:
    pip install "mediapipe==0.10.14" opencv-python scipy "numpy<2"
    (新しいmediapipeは mp.solutions が無くエラーになるため、バージョンを固定)
    仮想カメラ出力を使う場合: pip install pyvirtualcam と OBS Studio(26以降)のインストール
    test.jpg(貼り付けたい顔。正面・無表情・明るい写真)を同じフォルダに置く
操作:
    Esc で終了
"""
import time

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

# ---------- 設定 ----------
SOURCE_PATH = "test.jpg"       # 日本語パスはcv2.imreadで失敗するので半角英数字で
CAM_INDEX = 0
WIDTH, HEIGHT = 640, 480
SMOOTH = 0.5                 # ランドマーク平滑化 0=なし〜0.9=強い(強いほど遅れる)
USE_SEAMLESS = False         # Trueで境目が自然になるが重い
USE_VCAM = True              # Trueで仮想カメラ(OBS Virtual Camera)へ出力
VCAM_FPS = 30                # 仮想カメラの宣言fps(実際の出力は処理速度しだい)
VCAM_MIRROR = False          # False: 左右反転を戻して送る(相手からは普通のカメラ映像に見える)
# --------------------------

mp_fm = mp.solutions.face_mesh


def _indices(*connections):
    s = set()
    for conn in connections:
        for a, b in conn:
            s.update((a, b))
    return s


# 468点すべてだと重いので、輪郭・目・眉・口・鼻・頬の要所だけ使う(約150点)
IDX = np.array(sorted(
    _indices(
        mp_fm.FACEMESH_FACE_OVAL, mp_fm.FACEMESH_LIPS,
        mp_fm.FACEMESH_LEFT_EYE, mp_fm.FACEMESH_RIGHT_EYE,
        mp_fm.FACEMESH_LEFT_EYEBROW, mp_fm.FACEMESH_RIGHT_EYEBROW,
    )
    | {1, 2, 4, 5, 6, 19, 94, 97, 98, 129, 168, 195, 197, 326, 327, 358,  # 鼻
       50, 101, 123, 187, 205, 280, 330, 352, 411, 425,                    # 頬
       8, 9, 151}                                                          # 額〜眉間
))
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def get_points(results, w, h):
    """FaceMeshの結果から、使う点だけをピクセル座標で返す。顔なしならNone"""
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([(p.x, p.y) for p in lm], dtype=np.float32)[IDX]
    return pts * np.float32([w, h])


def rect_of(t):
    x0, y0 = np.floor(t.min(axis=0)).astype(int)
    x1, y1 = np.ceil(t.max(axis=0)).astype(int) + 1
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def prepare_source(img, pts):
    """元画像を三角形に分割し、切り出しを事前計算しておく(起動時に1回だけ)"""
    h, w = img.shape[:2]
    pts = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.float32)
    data = []
    for tri in Delaunay(pts).simplices:
        t = pts[tri]
        x, y, rw, rh = rect_of(t)
        crop = img[y:y + rh, x:x + rw].copy()
        data.append((tri, crop, (t - [x, y]).astype(np.float32)))
    return data


def warp_face(src_data, dst_pts, shape):
    """元画像の各三角形を、カメラの顔の対応する三角形へ変形して貼る"""
    H, W = shape[:2]
    canvas = np.zeros((H, W, 3), np.uint8)
    cover = np.zeros((H, W), np.uint8)
    for tri, crop, t1 in src_data:
        t2 = dst_pts[tri]
        x, y, w, h = rect_of(t2)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
            continue  # 画面外にはみ出す三角形は飛ばす
        t2r = (t2 - [x, y]).astype(np.float32)
        M = cv2.getAffineTransform(t1, t2r)
        patch = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, np.round(t2r).astype(np.int32), 255)
        np.copyto(canvas[y:y + h, x:x + w], patch, where=m[..., None] > 0)
        cover[y:y + h, x:x + w] |= m
    return canvas, cover


def match_color(src, ref, mask):
    """貼る顔の平均色を、カメラ側の顔の平均色に寄せる(LAB空間)"""
    sel = mask > 0
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    s += r[sel].mean(axis=0) - s[sel].mean(axis=0)
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite(frame, canvas, cover):
    """変形した顔を、色合わせ+ぼかしたマスクで元フレームに合成"""
    inner = cv2.erode(cover, KERNEL)
    ys, xs = np.nonzero(inner)
    if len(xs) < 200:
        return frame
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    f = frame[y0:y1, x0:x1]
    c = match_color(canvas[y0:y1, x0:x1], f, inner[y0:y1, x0:x1])
    m = inner[y0:y1, x0:x1]

    if USE_SEAMLESS:
        try:
            center = (int((x0 + x1) // 2), int((y0 + y1) // 2))
            return cv2.seamlessClone(c, frame, m, center, cv2.NORMAL_CLONE)
        except cv2.error:
            pass  # 失敗したら通常合成にフォールバック

    alpha = cv2.GaussianBlur(m, (21, 21), 0).astype(np.float32)[..., None] / 255.0
    out = frame.copy()
    out[y0:y1, x0:x1] = (f * (1 - alpha) + c * alpha).astype(np.uint8)
    return out


def detect_source(img):
    """元画像から顔ランドマークを取る。大きすぎ/顔が画像いっぱい等でも拾えるよう、
    リサイズと余白追加を変えて再試行する。成功したら(使った画像, 点)を返す"""
    h, w = img.shape[:2]
    if max(h, w) > 1280:                      # 巨大な写真は縮小
        k = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    for pad_ratio in (0.0, 0.3, 0.6):         # 顔が画像いっぱいの場合は余白を足して再試行
        p = int(max(img.shape[:2]) * pad_ratio)
        t = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE) if p else img
        with mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1,
                            min_detection_confidence=0.3) as fm:
            res = fm.process(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        pts = get_points(res, t.shape[1], t.shape[0])
        if pts is not None:
            return t, pts
    return img, None


def open_vcam(w, h):
    """仮想カメラを開く。失敗したらメッセージを出してNoneを返す(プレビューだけは動かす)"""
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=VCAM_FPS,
                                  fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"仮想カメラへ出力中: {cam.device}")
        return cam
    except ImportError:
        print("pyvirtualcam が未インストールです: pip install pyvirtualcam")
    except Exception as e:
        print(f"仮想カメラを開始できません: {e}\n"
              "OBS Studio が入っているか、OBS側で仮想カメラが起動中でないか確認してください。")
    return None


def main():
    src_img = cv2.imread(SOURCE_PATH)
    if src_img is None:
        raise SystemExit(f"{SOURCE_PATH} を読み込めません。パスとファイル名を確認してください。")

    print(f"元画像: {src_img.shape[1]}x{src_img.shape[0]}px")
    src_img, src_pts = detect_source(src_img)
    if src_pts is None:
        raise SystemExit(
            f"{SOURCE_PATH} から顔を検出できませんでした。\n"
            "顔が大きく写った正面向きの写真か、サングラス・マスク・強い影・イラストでないか確認してください。")
    src_data = prepare_source(src_img, src_pts)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    face_mesh = mp_fm.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    prev, fps, t_prev = None, 0.0, time.time()
    vcam, use_vcam = None, USE_VCAM

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 鏡表示
        h, w = frame.shape[:2]

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts = get_points(res, w, h)
        if pts is not None:
            if prev is not None and SMOOTH > 0:
                pts = SMOOTH * prev + (1 - SMOOTH) * pts
            prev = pts
            canvas, cover = warp_face(src_data, pts, frame.shape)
            frame = composite(frame, canvas, cover)
        else:
            prev = None

        # 仮想カメラへ送る(fps表示を描く前の、きれいな映像を送る)
        if use_vcam and vcam is None:
            vcam = open_vcam(w, h)
            use_vcam = vcam is not None
        if vcam is not None:
            vcam.send(frame if VCAM_MIRROR else cv2.flip(frame, 1))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"{fps:.1f} fps", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("face swap (mesh)", frame)
        if cv2.waitKey(1) == 27:  # Esc
            break

    cap.release()
    if vcam is not None:
        vcam.close()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
