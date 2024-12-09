# Qiita-AdventCalendar-20241210-OpenCV
Qiita OpenCV アドベントカレンダー(2024年12月10日)の投稿「[OpenCVのInpaintingでオクルージョン画像の物体検出精度向上🔍](https://qiita.com/Kazuhito/items/87b41542f71abd89cf62)」で使用したソースコードです。


# Create Mask
マスク画像の生成は以下です。<br>
Inpaintingしたい箇所をマウスで右クリックし囲んでください。<br>
```bash
python create_mask.py
```
<img src="https://github.com/user-attachments/assets/f4fba4ee-04c4-4c3d-9610-09a2e9a2a3bb" loading="lazy" width="45%">　<img src="https://github.com/user-attachments/assets/be93f468-3af6-463c-87fc-86ac709ed78d" loading="lazy" width="45%">

# Inpainting + Object Detection
Inpaintingを行い、物体検出を行うサンプルは以下です。
```bash
python test.py
```
![image](https://github.com/user-attachments/assets/45c07058-3ffe-4aad-b87b-84a9b1df7042)

# ProPainter
ProPainterのInpaintingを試したい方は、以下のノートブックをColaboratoryで上から順に実行してください。※A100必須<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Qiita-AdventCalendar-20241210-OpenCV//blob/main/Qiita-Advent-Calendar-ProPainter.ipynb)

# Authors
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Qiita-AdventCalendar-20241210-OpenCV is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[ドバイ（３）道路渋滞 アップ](https://www2.nhk.or.jp/archives/movies/?id=D0002050330_00000)を使用しています。
