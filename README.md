# faceswap

MediaPipe Face Mesh と三角形ワープによる、CPU だけで動く簡易リアルタイム顔入れ替えです。
Web カメラの映像に写った顔へ、用意した写真の顔を貼り付けて表示します。
OBS の仮想カメラへ出力すれば、Zoom や Teams などのビデオ会議でも使えます。

## 仕組み

1. 起動時に、貼り付けたい顔写真から顔のランドマーク(約150点)を検出し、Delaunay 三角形分割しておく
2. カメラの各フレームで同じランドマークを検出する
3. 写真の三角形を、カメラ側の対応する三角形へアフィン変換して貼る
4. LAB 色空間で肌の色をカメラ側に合わせ、ぼかしたマスクで合成する

GPU や学習済みの顔入れ替えモデルは使わないため、軽量ですが写真の表情は動きません(口や目の開閉は形の変形で追従します)。

## 動作環境

- Windows(カメラは DirectShow で開いています)
- Python 3.9〜3.12
- Web カメラ
- 仮想カメラ出力を使う場合: [OBS Studio](https://obsproject.com/) 26 以降

## セットアップ

プロジェクト専用の仮想環境(venv)に必要なパッケージだけを入れて、軽量に動かしています。
グローバルの Python 環境を汚さず、mediapipe などのバージョン固定もこのフォルダ内で完結します。

PowerShell で以下を実行します(例として `C:\faceswap` に置いた場合)。

```powershell
cd C:\faceswap
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> mediapipe は `0.10.14` に固定しています。新しいバージョンには `mp.solutions` がなく、エラーになります。
> 同じ理由で numpy も 2 未満に固定しています。

## 使い方

1. 貼り付けたい顔の写真を `test.jpg` という名前でこのフォルダに置きます
   - 正面向き・無表情・明るい写真が向いています
   - サングラス、マスク、強い影、イラストでは顔を検出できないことがあります
   - ファイル名・パスは半角英数字にしてください(`cv2.imread` は日本語パスを読めません)
2. PowerShell で venv の Python を直接指定して実行します(`activate` は不要です)

   ```powershell
   cd C:\faceswap
   .venv\Scripts\python.exe face_swap_mesh.py
   ```

3. プレビューウィンドウで `Esc` を押すと終了します

別の画像を使う場合は、`face_swap_mesh.py` の `SOURCE_PATH` を書き換えてください。

### カメラが映らないとき

どのカメラ番号で映像が取れるかを調べられます。

```powershell
.venv\Scripts\python.exe cam_test.py
```

「★映像あり」と出た番号を `face_swap_mesh.py` の `CAM_INDEX` に設定してください。

### ビデオ会議で使う(Google Meet など)

OBS Studio でプレビューウィンドウをキャプチャし、OBS の仮想カメラ経由で会議アプリに渡します。

1. `face_swap_mesh.py` を実行し、プレビューウィンドウ「face swap (mesh)」を表示しておきます
2. OBS Studio の「ソース」で「+」→「ウィンドウキャプチャ」を追加し、ウィンドウに
   `[python.exe]: face swap (mesh)` を選びます
3. 映像が画面いっぱいになるよう、ソースを右クリック →「変換」→「画面に合わせる」で調整します
4. OBS の「コントロール」で「仮想カメラ開始」を押します
5. Google Meet の「設定」→「動画」→「カメラ」で「OBS Virtual Camera」を選びます

> - プレビューは鏡表示(左右反転)です。相手に普通の向きで見せたい場合は、OBS でソースを右クリック →「変換」→「水平反転」してください
> - プレビュー左上の fps 表示もそのまま映ります。気になる場合は、ソースを Alt キーを押しながらドラッグしてトリミングしてください
> - この方法では OBS 側が仮想カメラを使うため、`USE_VCAM = False` にしておくと起動時の「仮想カメラを開始できません」の表示が出なくなります(`True` のままでもプレビューは動きます)

#### 別の方法: スクリプトから仮想カメラへ直接出力

`USE_VCAM = True` にすると、pyvirtualcam で OBS Virtual Camera へ直接映像を送ります。
この場合は OBS 本体で「仮想カメラ開始」を押さずに、OBS Studio をインストールしておくだけで使えます。
fps 表示の入らない映像が、左右反転を戻した向き(`VCAM_MIRROR = False` の場合)で送られます。

## 設定

`face_swap_mesh.py` 冒頭の定数で調整できます。

| 名前 | 既定値 | 説明 |
| --- | --- | --- |
| `SOURCE_PATH` | `"test.jpg"` | 貼り付ける顔写真 |
| `CAM_INDEX` | `0` | 使うカメラの番号 |
| `WIDTH`, `HEIGHT` | `640`, `480` | カメラの解像度 |
| `SMOOTH` | `0.5` | ランドマークの平滑化。0 でなし〜0.9 で強い(強いほど遅れる) |
| `USE_SEAMLESS` | `False` | `True` で境目が自然になるが重くなる |
| `USE_VCAM` | `True` | 仮想カメラへ出力する |
| `VCAM_FPS` | `30` | 仮想カメラに宣言する fps |
| `VCAM_MIRROR` | `False` | `False` なら左右反転を戻して送る(相手からは普通のカメラ映像に見える) |

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `face_swap_mesh.py` | 本体 |
| `cam_test.py` | カメラ診断ツール |
| `face_swap_mesh_old.py` | 旧バージョン |
| `requirements.txt` | 依存パッケージ |

画像ファイル(`*.jpg`, `*.png`, `*.webp` など)は `.gitignore` でリポジトリから除外しています。

## 注意

本人の同意なく他人の顔を使ったり、なりすましや人を欺く目的で使ったりしないでください。
