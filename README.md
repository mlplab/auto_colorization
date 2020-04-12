# Auto Colorization

お試しです．

実験設定，実験経過を記録

---

---

## 環境，使用するデータセット

- OS
  - Linux(Ubuntu16.04)
  - conda
- Module
  - python 3.6.10
  - tqdm 4.44.1
  - numpy 1.18.1
  - matplotlib 3.1.3
  - pillow 7.0.0
  - pytorch 1.4.0
  - torchvision 0.5.0
- Dataset
  - Food101

---

### 環境設定(Conda)

```
$ conda create -n [env_name] --file environment.yaml
```

---

---

## 進捗

### 4/8

- レポジトリ作成
- Encoder - Decoder 構造の着色ネットワークの作成
- UNet 構造の着色ネットワークの作成

---

### 4/11

- 学習時における progress bar の追加

---

### 4/12

- Dense-Unet の実装
