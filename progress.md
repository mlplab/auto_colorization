# 実験経過


## 4/12

- Unetの学習完了. 

    * 結果
        (画像，のちのち添付する予定)
        青色や緑色の再現が出来ていない.
    
    * なぜ？

        1. 使用したデータ形式が原因？

            今回の学習ではグレースケールの画像を入力とし，RGB画像を出力として学習した．

            $\rightarrow$ 色合いの再現にRGBえは情報量が少なすぎる...?

            $\rightarrow$ 使用するデータ形式をする必要がある(色合いではなく輝度値とか)...?

        2. UNet を使用した

            グレースケール画像の畳み込み特徴とかはskip connectionで引き継げているはず...

            $\rightarrow$ 空間的特徴(Spartial feature)は引き継げている．実際に学習開始時に画像の輪郭は正確にとらえられていた．

            $\Rightarrow$ チャンネル特徴は...?

            $\rightarrow$ チャンネル特徴がうまいこと引き継げていない

            $\rightarrow$ Channel Attentionとかやってみる...?

        
    * 今後

        1. データ形式の変更(明るさを表現するYCrCb画像，赤っぽい色と青っぽい色と輝度値で表現されるLAB画像) を検討．
        2. Channel Attention(SENet)やSeparable Attention
        の導入を検討．
